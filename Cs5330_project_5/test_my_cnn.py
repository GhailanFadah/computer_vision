"""
Ghailan Fadah
CS5530
4/6/26
loads the model saved from my_cnn and evaluates it on n images from test set. 
also tests the model on my own created dataset 
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from my_cnn import DigitRecognitionNet


def load_model(path='model.pth'):
    """Load the saved model weights into a DigitRecognitionNet."""
    model = DigitRecognitionNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


class MyDigitsDataset(Dataset):
    """Loads hand-written digit images from a flat folder, transforms them to match MNIST."""

    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.Lambda(lambda x: TF.invert(x)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image), os.path.basename(self.paths[idx])


def get_my_digits_loader(digits_dir='./data/my_digits'):
    """Return a DataLoader for the custom hand-written digit images."""
    dataset = MyDigitsDataset(digits_dir)
    return DataLoader(dataset, batch_size=len(dataset), shuffle=False)


def get_test_loader(data_dir='./data', batch_size=10):
    """Return a DataLoader for the MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_first_n(model, test_loader, n=10):
    """
    Run the model on the first n examples.
    Prints output values, predicted label, and correct label for each.
    Returns the images, predictions, and labels.
    """
    images, labels = next(iter(test_loader))
    images, labels = images[:n], labels[:n]

    with torch.no_grad():
        outputs = model(images)

    # outputs are log-softmax; exponentiate to get probabilities
    probs = outputs.exp()

    for i in range(n):
        values = '  '.join(f'{v:.2f}' for v in probs[i])
        predicted = probs[i].argmax().item()
        correct = labels[i].item()
        print(f'Sample {i+1}: [{values}]  pred={predicted}  label={correct}')

    return images, probs.argmax(dim=1).tolist(), labels.tolist()


def plot_predictions(images, predictions, n=9):
    """Plot the first n examples in a 3x3 grid with the prediction above each image."""
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {predictions[i]}', fontsize=12)
        axes[i].axis('off')

    fig.suptitle('First 9 Test Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()


def evaluate_my_digits(model, loader):
    """Run the model on all images in the custom digits loader and print results."""
    images, filenames = next(iter(loader))

    with torch.no_grad():
        probs = model(images).exp()

    for i in range(len(images)):
        values = '  '.join(f'{v:.2f}' for v in probs[i])
        predicted = probs[i].argmax().item()
        print(f'{filenames[i]}: [{values}]  pred={predicted}')

    predictions = probs.argmax(dim=1).tolist()
    return images, predictions


def main():
    model = load_model('model.pth')

    # MNIST test set
    test_loader = get_test_loader(batch_size=10)
    images, predictions, _ = evaluate_first_n(model, test_loader, n=10)
    plot_predictions(images, predictions, n=9)

    # Custom hand-written digits
    print('\n--- My Digits ---')
    my_loader = get_my_digits_loader('./data/my_digits')
    my_images, my_predictions = evaluate_my_digits(model, my_loader)
    plot_predictions(my_images, my_predictions, n=min(9, len(my_images)))


if __name__ == '__main__':
    main()
