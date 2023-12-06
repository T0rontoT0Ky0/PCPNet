from __future__ import print_function
import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter
import utils
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from Arch import PCPNet
from train_loader import parse_arguments
from Loss import compute_loss

def train_pcpnet(opt):

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # create model
    if len(opt.patch_radius) == 1:
        pcpnet = PCPNet(
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)


    if opt.refine != '':
        pcpnet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass


    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    optimizer = optim.SGD(pcpnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1) # milestones in number of optimizer iterations
    pcpnet.to(device)

    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(opt.nepoch):

        train_batchind = -1
        train_fraction_done = 0.0
        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, data in train_enum:

            # update learning rate
            scheduler.step(epoch * train_num_batch + train_batchind)

            # set to training mode
            pcpnet.train()

            # get trainingset batch and upload to GPU
            points = data[0]
            target = data[1:-1]

            points = points.transpose(2, 1)
            points = points.to(device)

            target = tuple(t.to(device) for t in target)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            pred, trans, _, _ = pcpnet(points)

            loss = compute_loss(
                pred=pred, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight,
                patch_rot=trans if opt.use_point_stn else None,
                normal_loss=opt.normal_loss)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            print('[%s %d: %d/%d] %s loss: %f' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'), loss.item()))
            train_writer.add_scalar('loss', loss.item(), (epoch + train_fraction_done) * train_num_batch * opt.batchSize)

            while test_fraction_done <= train_fraction_done and test_batchind+1 < test_num_batch:

                # set to evaluation mode
                pcpnet.eval()

                test_batchind, data = next(test_enum)

                # get testset batch and upload to GPU
                points = data[0]
                target = data[1:-1]

                points = points.transpose(2, 1)
                points = points.to(device)

                target = tuple(t.to(device) for t in target)

                # forward pass
                with torch.no_grad():
                    pred, trans, _, _ = pcpnet(points)

                loss = compute_loss(
                    pred=pred, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight,
                    patch_rot=trans if opt.use_point_stn else None,
                    normal_loss=opt.normal_loss)

                test_fraction_done = (test_batchind+1) / test_num_batch

                # print info and update log file
                print('[%s %d: %d/%d] %s loss: %f' % (opt.name, epoch, train_batchind, train_num_batch-1, blue('test'), loss.item()))
                test_writer.add_scalar('loss', loss.item(), (epoch + test_fraction_done) * train_num_batch * opt.batchSize)

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            torch.save(pcpnet.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(pcpnet.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
