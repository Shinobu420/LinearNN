# Linear Neural Networks

This training exercise demonstrates training and utilizing Linear Neural Networks using PyTorch and TorchVision.

## Applications

*   **[linearNN.py](linearNN.py)**: Defines the [neuralNetwork](linearNN.py#L5) structure and the data transformation [transformData](linearNN.py#L25).
*   **[main.py](main.py)**: Trains the model on images in `./PetImages/` and saves the weights to `cat_dog_model.pth`.
*   **[test.py](test.py)**: Loads `cat_dog_model.pth` and predicts the class of a local image `image.png`.


## Usage

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pillow
   ```

2. **Prepare Dataset**:
   Organize your dataset under `./PetImages` (e.g., `./PetImages/Cat/` and `./PetImages/Dog/`). Having an even split is recommended

3. **Train**:
   ```bash
   python main.py
   ```

4. **Test**:
   Place an image named `image.png` in the project root directory and run:
   ```bash
   python test.py
   ```
