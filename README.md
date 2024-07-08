# ASL Image Classification with EfficientNet

This project involves classifying American Sign Language (ASL) images using a pre-trained EfficientNet-B0 model. The project demonstrates transfer learning techniques and utilizes the PyTorch framework along with torchvision for model and data handling.

## Project Structure

- `ASL_Classification.ipynb`: Jupyter notebook containing the entire workflow from data loading to model training and evaluation.
- `models/`: Directory where the trained models are saved.
- `data/`: Directory containing the dataset.
- `README.md`: Documentation for the project.

## Dataset

The dataset contains images of ASL signs, divided into training and testing directories. Each subdirectory within `train` and `test` corresponds to a different ASL sign.

- `data/train/`: Training images
- `data/test/`: Testing images

## Preprocessing

The preprocessing steps include:

1. Loading the dataset from the specified directories.
2. Applying the appropriate transformations for EfficientNet-B0 using the pre-trained weights.
3. Creating data loaders for training and testing.

## Model

The model is based on EfficientNet-B0, a state-of-the-art architecture for image classification. The following steps are involved:

1. Loading the pre-trained EfficientNet-B0 model with default weights.
2. Freezing the feature extraction layers.
3. Replacing the classifier head with a custom classifier suited for the ASL dataset.

## Training

The training process involves:

1. Defining the loss function (CrossEntropyLoss) and optimizer (SGD).
2. Training the model for a specified number of epochs (5 in this case).
3. Recording the training and testing losses and accuracies for each epoch.
4. Logging the training process using TensorBoard.

## Evaluation

The model's performance is evaluated by printing the training and testing accuracies and losses over the epochs. The total training time is also recorded.

## Dependencies

- torch
- torchvision
- matplotlib
- torchinfo
- TensorBoard

## Results

The training and testing losses and accuracies are recorded and can be visualized using TensorBoard. The final trained model is saved in the `models` directory.

## Conclusion

This project demonstrates how to use transfer learning with EfficientNet-B0 to classify ASL images. Further improvements can be made by tuning the model architecture, experimenting with different loss functions and optimizers, and using more advanced data augmentation techniques.
