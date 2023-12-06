def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):

    loss = 0

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]

            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-torch.abs(utils.cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred-o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return loss