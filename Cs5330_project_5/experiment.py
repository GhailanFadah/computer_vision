"""
Ghailan Fadah
CS5530
4/6/26
explores hyperparameter tuning on the Fashion MNIST dataset
Strategy: linear search (optimize one dimension at a time, round-robin across
dimensions), then a random search phase. Target: ~60 total experiments.

Results are saved to experiment_results.csv and plotted at the end.
"""

import csv
import time
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


RESULTS_FILE = 'experiment_results.csv'
DATA_DIR = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline configuration
BASELINE = dict(
    conv1_filters=10,
    conv2_filters=20,
    fc_hidden=50,
    dropout=0.5,
    epochs=5,
    batch_size=64,
    learning_rate=0.01,
    momentum=0.5,
)


@dataclass
class ExpConfig:
    conv1_filters: int = 10
    conv2_filters: int = 20
    fc_hidden: int = 50
    dropout: float = 0.5
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.5
    label: str = 'baseline'


class FlexibleCNN(nn.Module):
    """
    Configurable CNN matching the DigitRecognitionNet architecture but with
    tunable filter counts and FC size for experimentation.

    Architecture:
      Conv(1->conv1_filters, 5x5) -> MaxPool(2x2) -> ReLU
      Conv(conv1_filters->conv2_filters, 5x5) -> Dropout -> MaxPool(2x2) -> ReLU
      Flatten -> FC(flat->fc_hidden) -> ReLU
      FC(fc_hidden->10) -> LogSoftmax
    """

    def __init__(self, config: ExpConfig, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(config.conv1_filters, config.conv2_filters, kernel_size=5)
        self.dropout = nn.Dropout2d(p=config.dropout)
        # 28x28 -> conv(5x5)->24x24 -> pool->12x12 -> conv(5x5)->8x8 -> pool->4x4
        self.flat_size = 4 * 4 * config.conv2_filters
        self.fc1 = nn.Linear(self.flat_size, config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_loaders_cache: dict = {}


def get_loaders(batch_size: int):
    """Return cached train/test DataLoaders for Fashion MNIST."""
    if batch_size not in _loaders_cache:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_loader = DataLoader(
            datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform),
            batch_size=1000, shuffle=False
        )
        _loaders_cache[batch_size] = (train_loader, test_loader)
    return _loaders_cache[batch_size]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(config: ExpConfig) -> dict:
    """Train a FlexibleCNN with the given config and return result metrics."""
    train_loader, test_loader = get_loaders(config.batch_size)

    model = FlexibleCNN(config).to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            F.nll_loss(model(data), target).backward()
            optimizer.step()

    # Final evaluation
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += output.argmax(dim=1).eq(target).sum().item()

    n = len(test_loader.dataset)
    elapsed = time.time() - start

    result = asdict(config)
    result['test_loss'] = round(test_loss / n, 4)
    result['test_acc'] = round(100. * correct / n, 2)
    result['train_time_s'] = round(elapsed, 1)
    return result


# ---------------------------------------------------------------------------
# Experiment plan
# ---------------------------------------------------------------------------

def build_experiment_plan() -> list[ExpConfig]:
    """
    Build ~60 configurations using a linear search strategy:
      Phase 1 – sweep each dimension independently from the baseline.
      Phase 2 – random search over all dimensions to catch interactions.
    """
    configs = []

    def make(**kwargs):
        params = {**BASELINE, **kwargs}
        return ExpConfig(**params)

    # --- Phase 1: Baseline ---
    configs.append(make(label='baseline'))

    # --- Phase 1a: Sweep conv1_filters ---
    for v in [4, 8, 16, 24, 32, 48]:
        configs.append(make(conv1_filters=v, label=f'conv1_filters={v}'))

    # --- Phase 1b: Sweep conv2_filters ---
    for v in [8, 16, 32, 40, 64]:
        configs.append(make(conv2_filters=v, label=f'conv2_filters={v}'))

    # --- Phase 1c: Sweep fc_hidden ---
    for v in [25, 75, 100, 150, 200, 300]:
        configs.append(make(fc_hidden=v, label=f'fc_hidden={v}'))

    # --- Phase 1d: Sweep dropout ---
    for v in [0.1, 0.2, 0.3, 0.4, 0.6, 0.75]:
        configs.append(make(dropout=v, label=f'dropout={v}'))

    # --- Phase 1e: Sweep batch_size ---
    for v in [16, 32, 128, 256, 512]:
        configs.append(make(batch_size=v, label=f'batch_size={v}'))

    # --- Phase 1f: Sweep learning_rate ---
    for v in [0.001, 0.005, 0.02, 0.05, 0.1]:
        configs.append(make(learning_rate=v, label=f'lr={v}'))

    # --- Phase 1g: Sweep epochs ---
    for v in [2, 3, 8, 10]:
        configs.append(make(epochs=v, label=f'epochs={v}'))

    # --- Phase 2: Random search ---
    random.seed(42)
    conv1_options = [8, 10, 16, 24, 32]
    conv2_options = [16, 20, 32, 40]
    fc_options = [50, 100, 150, 200]
    dropout_options = [0.2, 0.3, 0.4, 0.5]
    batch_options = [32, 64, 128]
    lr_options = [0.005, 0.01, 0.02]

    for i in range(25):
        configs.append(make(
            conv1_filters=random.choice(conv1_options),
            conv2_filters=random.choice(conv2_options),
            fc_hidden=random.choice(fc_options),
            dropout=random.choice(dropout_options),
            batch_size=random.choice(batch_options),
            learning_rate=random.choice(lr_options),
            label=f'random_{i}'
        ))

    return configs


# ---------------------------------------------------------------------------
# Results I/O and plotting
# ---------------------------------------------------------------------------

def save_result(result: dict, first: bool = False):
    """Append a result row to the CSV file."""
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if first:
            writer.writeheader()
        writer.writerow(result)


def load_results() -> list[dict]:
    """Read all results from the CSV file."""
    with open(RESULTS_FILE, newline='') as f:
        return list(csv.DictReader(f))


def plot_dimension(results: list[dict], dim: str, title: str):
    """Plot test accuracy vs a single swept dimension (linear search results only)."""
    rows = [r for r in results if r['label'].startswith(dim) or r['label'] == 'baseline']
    if not rows:
        return

    xs = [float(r[dim]) for r in rows]
    ys = [float(r['test_acc']) for r in rows]
    pairs = sorted(zip(xs, ys))
    xs, ys = zip(*pairs)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(dim)
    plt.ylabel('Test Accuracy (%)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_results(results: list[dict]):
    """Summary plots for each swept dimension and an overview scatter."""
    dims = [
        ('conv1_filters', 'Effect of conv1 filter count'),
        ('conv2_filters', 'Effect of conv2 filter count'),
        ('fc_hidden',     'Effect of FC hidden size'),
        ('dropout',       'Effect of dropout rate'),
        ('batch_size',    'Effect of batch size'),
        ('learning_rate', 'Effect of learning rate'),
        ('epochs',        'Effect of training epochs'),
    ]
    for dim, title in dims:
        plot_dimension(results, dim, title)

    # Scatter: all experiments ranked by accuracy
    accs = sorted([float(r['test_acc']) for r in results])
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(accs)), accs, s=15, alpha=0.7)
    plt.axhline(accs[0], color='r', linestyle='--', label=f'worst: {accs[0]:.2f}%')
    plt.axhline(accs[-1], color='g', linestyle='--', label=f'best: {accs[-1]:.2f}%')
    plt.xlabel('Experiment rank')
    plt.ylabel('Test Accuracy (%)')
    plt.title('All Experiments — Accuracy Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Using device: {DEVICE}')
    print('Dataset: Fashion MNIST\n')

    configs = build_experiment_plan()
    print(f'Total experiments planned: {len(configs)}\n')

    results = []
    for i, cfg in enumerate(configs):
        print(f'[{i+1:3d}/{len(configs)}] {cfg.label} ... ', end='', flush=True)
        result = train_and_evaluate(cfg)
        results.append(result)
        save_result(result, first=(i == 0))
        print(f'acc={result["test_acc"]:.2f}%  loss={result["test_loss"]:.4f}  '
              f'time={result["train_time_s"]:.0f}s')

    # Summary
    best = max(results, key=lambda r: r['test_acc'])
    worst = min(results, key=lambda r: r['test_acc'])
    print(f'\nBest:  {best["test_acc"]:.2f}%  ({best["label"]})')
    print(f'Worst: {worst["test_acc"]:.2f}%  ({worst["label"]})')
    print(f'Results saved to {RESULTS_FILE}')

    plot_all_results(results)


if __name__ == '__main__':
    main()
