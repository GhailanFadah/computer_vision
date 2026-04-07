"""
Ghailan Fadah
CS5530
4/6/26
uses PyTorch framework to create a simple CNN to recongize digits using the
MNIST dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DigitRecognitionNet(nn.Module):
    """
    CNN for MNIST digit recognition.
    Architecture:
      1. Conv(1->10, 5x5)
      2. MaxPool(2x2) + ReLU
      3. Conv(10->20, 5x5)
      4. Dropout(p=0.5)
      5. MaxPool(2x2) + ReLU
      6. Flatten -> FC(320->50) -> ReLU
      7. FC(50->10) -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Conv -> MaxPool -> ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv -> Dropout -> MaxPool -> ReLU
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # Flatten
        x = x.view(-1, 320)
        # FC -> ReLU
        x = F.relu(self.fc1(x))
        # FC -> LogSoftmax
        return F.log_softmax(self.fc2(x), dim=1)


def get_data_loaders(batch_size_train=64, batch_size_test=1000, data_dir='./data'):
    """Download MNIST and return train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size_train,
        shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=batch_size_test,
        shuffle=False
    )
    return train_loader, test_loader


def plot_test_samples(test_loader, n=6):
    """Display the first N samples from the test set in a grid."""
    images, labels = next(iter(test_loader))
    images, labels = images[:n], labels[:n]

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'First {n} Test Samples', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_accuracy(train_accuracies, test_accuracies):
    """Plot train and test accuracy over epochs."""
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(train_losses, test_losses):
    """Plot train and test loss over epochs."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Run one training epoch, print progress, and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += F.nll_loss(output, target, reduction='sum').item()
        correct += output.argmax(dim=1).eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f'Train Epoch {epoch} '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]  '
                f'Loss: {loss.item():.6f}'
            )

    n = len(train_loader.dataset)
    return total_loss / n, 100. * correct / n


def test(model, device, test_loader):
    """Evaluate the model on the test set, print results, and return (avg_loss, accuracy)."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    n = len(test_loader.dataset)
    test_loss /= n
    accuracy = 100. * correct / n
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{n} ({accuracy:.2f}%)\n'
    )
    return test_loss, accuracy


def save_model(model, path='model.pth'):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def train_and_save(epochs=5, learning_rate=0.01, momentum=0.5, batch_size_train=64,
                   batch_size_test=1000, model_path='model.pth', data_dir='./data',):
    """Full pipeline: build, train, evaluate, and save the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = get_data_loaders(batch_size_train, batch_size_test, data_dir)

    plot_test_samples(test_loader, n=6)

    model = DigitRecognitionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plot_loss(train_losses, test_losses)
    save_model(model, model_path)
    return model


if __name__ == '__main__':
    train_and_save()
