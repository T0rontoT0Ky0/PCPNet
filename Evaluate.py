from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from dataset import PointcloudPatchDataset, SequentialPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from Arch import PCPNet
from eval_loader import parse_arguments


def evaluate(opt):

    opt.models = opt.models.split()

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    for model_name in opt.models:

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        model_filename = os.path.join(opt.modeldir, model_name+opt.modelpostfix)
        param_filename = os.path.join(opt.modeldir, model_name+opt.parmpostfix)

        # load model and training parameters
        trainopt = torch.load(param_filename)

        if opt.batchSize == 0:
            model_batchSize = trainopt.batchSize
        else:
            model_batchSize = opt.batchSize

        # get indices in targets and predictions corresponding to each output
        pred_dim = 0
        output_pred_ind = []
        for o in trainopt.outputs:
            if o == 'unoriented_normals' or o == 'oriented_normals':
                output_pred_ind.append(pred_dim)
                pred_dim += 3
            elif o == 'max_curvature' or o == 'min_curvature':
                output_pred_ind.append(pred_dim)
                pred_dim += 1
            else:
                raise ValueError('Unknown output: %s' % (o))

        dataset = PointcloudPatchDataset(
            root=opt.indir, shape_list_filename=opt.dataset,
            patch_radius=trainopt.patch_radius,
            points_per_patch=trainopt.points_per_patch,
            patch_features=[],
            seed=opt.seed,
            use_pca=trainopt.use_pca,
            center=trainopt.patch_center,
            point_tuple=trainopt.point_tuple,
            sparse_patches=opt.sparse_patches,
            cache_capacity=opt.cache_capacity)
        if opt.sampling == 'full':
            datasampler = SequentialPointcloudPatchSampler(dataset)
        elif opt.sampling == 'sequential_shapes_random_patches':
            datasampler = SequentialShapeRandomPointcloudPatchSampler(
                dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                sequential_shapes=True,
                identical_epochs=False)
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=datasampler,
            batch_size=model_batchSize,
            num_workers=int(opt.workers))

        if len(trainopt.patch_radius) == 1:
            regressor = PCPNet(
                num_points=trainopt.points_per_patch,
                output_dim=pred_dim,
                use_point_stn=trainopt.use_point_stn,
                use_feat_stn=trainopt.use_feat_stn,
                sym_op=trainopt.sym_op,
                point_tuple=trainopt.point_tuple)

        regressor.load_state_dict(torch.load(model_filename))
        regressor.to(device)
        regressor.eval()

        shape_ind = 0
        shape_patch_offset = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        shape_properties = torch.zeros(shape_patch_count, pred_dim, dtype=torch.float, device=device)

        # append model name to output directory and create directory if necessary
        model_outdir = os.path.join(opt.outdir, model_name)
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)
        for batchind, data in batch_enum:

            # get batch and upload to GPU
            points, data_trans = data
            points = points.transpose(2, 1)
            points = points.to(device)

            data_trans = data_trans.to(device)

            with torch.no_grad():
                pred, trans, _, _ = regressor(points)

            # post-processing of the prediction
            for oi, o in enumerate(trainopt.outputs):
                if o == 'unoriented_normals' or o == 'oriented_normals':
                    o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]

                    if trainopt.use_point_stn:
                        # transform predictions with inverse transform
                        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                        o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)

                    if trainopt.use_pca:
                        # transform predictions with inverse pca rotation (back to world space)
                        o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

                    # normalize normals
                    o_pred_len = torch.max(o_pred.new_tensor([sys.float_info.epsilon*100]), o_pred.norm(p=2, dim=1, keepdim=True))
                    o_pred = o_pred / o_pred_len

                elif o == 'max_curvature' or o == 'min_curvature':
                    o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]

                    # undo patch size normalization:
                    o_pred[:, :] = o_pred / dataset.patch_radius_absolute[shape_ind][0]

                else:
                    raise ValueError('Unsupported output type: %s' % (o))

            print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))

            batch_offset = 0
            while batch_offset < pred.size(0):

                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = pred.size(0)-batch_offset

                # append estimated patch properties batch to properties for the current shape
                shape_properties[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = pred[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:

                    # save shape properties to disk
                    prop_saved = [False]*len(trainopt.outputs)

                    # save normals
                    oi = [i for i, o in enumerate(trainopt.outputs) if o in ['unoriented_normals', 'oriented_normals']]
                    if len(oi) > 1:
                        raise ValueError('Duplicate normal output.')
                    elif len(oi) == 1:
                        oi = oi[0]
                        normal_prop = shape_properties[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.normals'), normal_prop.cpu().numpy())
                        prop_saved[oi] = True

                    # save curvatures
                    oi1 = [i for i, o in enumerate(trainopt.outputs) if o == 'max_curvature']
                    oi2 = [i for i, o in enumerate(trainopt.outputs) if o == 'min_curvature']
                    if len(oi1) > 1 or len(oi2) > 1:
                        raise ValueError('Duplicate minimum or maximum curvature output.')
                    elif len(oi1) == 1 or len(oi2) == 1:
                        curv_prop = shape_properties.new_zeros(shape_properties.size(0), 2)
                        if len(oi1) == 1:
                            oi1 = oi1[0]
                            curv_prop[:, 0] = shape_properties[:, output_pred_ind[oi1]]
                            prop_saved[oi1] = True
                        if len(oi2) == 1:
                            oi2 = oi2[0]
                            curv_prop[:, 1] = shape_properties[:, output_pred_ind[oi2]]
                            prop_saved[oi2] = True
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.curv'), curv_prop.cpu().numpy())

                    if not all(prop_saved):
                        raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')

                    # save point indices
                    if opt.sampling != 'full':
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.idx'), datasampler.shape_patch_inds[shape_ind], fmt='%d')

                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif opt.sampling == 'sequential_shapes_random_patches':
                            # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                        shape_properties = shape_properties.new_zeros(shape_patch_count, pred_dim)


if __name__ == '__main__':
    eval_opt = parse_arguments()
    evaluate(eval_opt)
