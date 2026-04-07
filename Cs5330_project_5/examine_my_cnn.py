"""
Ghailan Fadah
CS5530
4/6/26
explores the CNN created in my_cnn:
Loads the trained model and prints its architecture
Visualizes and prints the 10 filters with pyplot
Applies filters to the first training image and plots result
"""

import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from my_cnn import DigitRecognitionNet


def load_model(path='model.pth'):
    """Load the saved model and set it to evaluation mode."""
    model = DigitRecognitionNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def get_first_training_image(data_dir='./data'):
    """Return the first image from the MNIST training set as a numpy array (H x W)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    image, _ = next(iter(loader))
    # shape: (1, 1, 28, 28) -> (28, 28) numpy float32
    return image.squeeze().numpy()


def print_filter_weights(model):
    """Print the weights and shape of the first conv layer (conv1)."""
    with torch.no_grad():
        weights = model.conv1.weight
    print(f'conv1 weight shape: {weights.shape}')
    print('conv1 filter weights:')
    print(weights)
    return weights


def plot_filters(weights):
    """Visualize the 10 filters of the first conv layer in a 2-row, 5-column grid."""
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('First Layer Filters (conv1)', fontsize=14)

    with torch.no_grad():
        filters = weights.numpy()  # shape: (10, 1, 5, 5)

    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(filters[i, 0], cmap='gray')
        ax.set_title(f'Filter {i}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def apply_filters_and_plot(image, weights):
    """Apply the 10 conv1 filters to the image using cv2.filter2D and plot results."""
    with torch.no_grad():
        filters = weights.numpy()  # shape: (10, 1, 5, 5)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('Images Filtered by conv1 Weights', fontsize=14)

    for i in range(10):
        kernel = filters[i, 0]  # (5, 5)
        filtered = cv2.filter2D(image, -1, kernel)

        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(filtered, cmap='gray')
        ax.set_title(f'Filter {i}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    model = load_model('model.pth')

    print('Model architecture:')
    print(model)
    print()

    weights = print_filter_weights(model)
    plot_filters(weights)

    image = get_first_training_image()
    apply_filters_and_plot(image, weights)


if __name__ == '__main__':
    main()
