
# PCPNet

## Python Files Description

This project includes the following Python files:

1. **Arch.py**: Structure of PCPNet.
2. **eval_loader.py**: Configuration for evaluation.
3. **Evaluate.py**: Evaluation function.
4. **Loss.py**: Loss function implementation.
5. **Train.py**: Model training function.
6. **train_loader.py**: Training configuration.
7. **Vision.py**: Visualization file.

## Training Steps

To train and evaluate using this project, follow these steps:

1. **Start Training**:
    - Run `python Train.py` to start training.
    - Trained models will be saved in the `models` folder.
    - You can select the type of training you want (normals or curvature) in `train_loader.py`.

2. **Evaluate the Model**:
    - Use `python Evaluate.py` to evaluate the trained model.
    - You can choose the training model `.pth` file and parameter file in `eval_loader.py`.

3. **Visualization**:
    - Run `Vision.py` for visualization.
    - Remember to select the correct path you want to visualize in the function(The path I put in the `Vision.py` has already told the file you should choose to visualize the training model. However the path is just an example, you must choose your path of those file!).
