import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to normalize a 1D array for color mapping
def normalize(array):
    norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    return norm

# Function to read point cloud data from a file
def load_point_cloud(file_path):
    return np.loadtxt(file_path)

# Function to check the consistency of point cloud data and attributes
def check_consistency(point_cloud, attributes):
    if point_cloud.shape[0] != attributes.shape[0]:
        raise ValueError(f"Point cloud data has {point_cloud.shape[0]} points but attributes have {attributes.shape[0]} entries.")

# Function to create and save the 3D scatter plot
def plot_and_save(point_cloud, attribute, colormap, title, save_path):
    # Check for consistency between the point cloud and the attribute
    check_consistency(point_cloud, attribute)
    
    # Normalize the attribute if it's not already a color map
    if attribute.ndim > 1:
        # Assume attribute is normals or similar and convert to single scalar per point
        attribute = np.linalg.norm(attribute, axis=1)
    color_map = colormap(normalize(attribute))

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=color_map, s=0.5)
    ax.set_title(title)
    ax.set_axis_off()

    # Save the figure
    plt.savefig(save_path)

    # Close the figure
    plt.close(fig)




def read_testset(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def process_testset(testset_file, dataset_dir, results_dir, output_dir):
    testset = read_testset(testset_file)

    for object_name in testset:
        point_cloud_path = f"{dataset_dir}/{object_name}.xyz"
        normals_path = f"{results_dir}/{object_name}.normals"
        curv_path = f"{results_dir}/{object_name}.curv"

        point_cloud = load_point_cloud(point_cloud_path)
        normals = load_point_cloud(normals_path)
        curvatures = load_point_cloud(curv_path)


        plot_and_save(point_cloud, point_cloud, plt.cm.gray, f'{object_name} - Noisy Input',f'{output_dir}/{object_name}_input.png')
        plot_and_save(point_cloud, normals, plt.cm.viridis, f'{object_name} - Normals',f'{output_dir}/{object_name}_normals.png')
        plot_and_save(point_cloud, curvatures, plt.cm.cividis, f'{object_name} - Curvature',f'{output_dir}/{object_name}_curvature.png')


testset_file_path = '/pclouds/testset_no_noise.txt'#path to 'testset_no_noise.txt'
dataset_directory = '/pclouds'#path to your download dataset of the PCPNet's author
results_directory = '/results/my_single_scale_normal_curv'#path to your own results after evaluate the traning model
output_directory = '/plot/my_train'#path to where you want uotput the image

process_testset(testset_file_path, dataset_directory, results_directory, output_directory)
