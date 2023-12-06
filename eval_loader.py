import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', type=str, default='testset_no_noise.txt', help='shape set file name')
    parser.add_argument('--modeldir', type=str, default='./models', help='model folder')
    parser.add_argument('--models', type=str, default='my_single_scale_normal', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model_1999.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    parser.add_argument('--sparse_patches', type=int, default=False, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

    return parser.parse_args()
