"""
Ghailan Fadah
CS5530
4/6/26
explores the concept of transfer learning:
Build and load the pre-trained MNIST network
Replace the last layer (fc2) with a new Linear(50, 3) for alpha/beta/gamma
Train only the new layer on the Greek letter dataset
evalulate the model both on test set as well as my own created dataset 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from my_cnn import DigitRecognitionNet


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


MY_GREEK_PATH = 'data/my_greek'



class MyGreekDataset(Dataset):
    """Loads custom Greek letter images from a flat folder using the GreekTransform."""

    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image), os.path.basename(self.paths[idx])


def get_my_greek_loader(folder='data/my_greek'):
    """Return a DataLoader for custom hand-written Greek letter images."""
    dataset = MyGreekDataset(folder)
    return DataLoader(dataset, batch_size=len(dataset), shuffle=False)


def evaluate_my_greek(model, loader):
    """Run the model on all custom Greek images and print predictions."""
    model.eval()
    images, filenames = next(iter(loader))

    with torch.no_grad():
        probs = model(images).exp()

    greek_classes = ['alpha', 'beta', 'gamma']
    print('\n--- My Greek Letters ---')
    for i in range(len(images)):
        values = '  '.join(f'{v:.2f}' for v in probs[i])
        pred_idx = probs[i].argmax().item()
        print(f'{filenames[i]}: [{values}]  pred={greek_classes[pred_idx]}')

    return images, probs.argmax(dim=1).tolist()


def load_pretrained_mnist_model(path='model.pth'):
    """Load the pre-trained MNIST model from disk."""
    model = DigitRecognitionNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def build_greek_model(mnist_model_path='model.pth'):
    """
    Build a transfer-learning model for Greek letters:
      - Load the pre-trained MNIST weights
      - Freeze all layers
      - Replace fc2 (the last layer) with Linear(50, 3)
    """
    model = load_pretrained_mnist_model(mnist_model_path)

    # Freeze every parameter so only the new layer will train
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer with one that has 3 output nodes
    model.fc2 = nn.Linear(50, 3)

    return model


def get_greek_data_loader(training_set_path='data/greek_train', batch_size=5):
    """Create a DataLoader for the Greek letter dataset using ImageFolder."""
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def train(model, train_loader, epochs=5, learning_rate=0.01, momentum=0.5):
    """Train the model and return the per-epoch average loss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch:3d}  Loss: {avg_loss:.6f}  Accuracy: {correct}/{total} ({accuracy:.1f}%)')

    return epoch_losses


def plot_loss(epoch_losses):
    """Plot the training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Greek Letter Transfer Learning — Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    model = build_greek_model('model.pth')
    train_loader = get_greek_data_loader('data/greek_train')

    print('Modified network architecture:')
    print(model)
    print(f'\nTraining on Greek letters from: {'data/greek_train'}')
    print(f'Classes: {train_loader.dataset.classes}\n')

    epoch_losses = train(model, train_loader, epochs=10)
    plot_loss(epoch_losses)

    my_loader = get_my_greek_loader('data/my_greek')
    evaluate_my_greek(model, my_loader)


if __name__ == '__main__':
    main()