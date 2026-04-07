# CS5330 Project 5 — Deep Learning with PyTorch

**Ghailan Fadah | CS5330 | Spring 2026**

A series of deep learning tasks using PyTorch and torchvision, covering digit recognition, transfer learning, network analysis, transformer-based classification, hyperparameter experimentation, and live inference.

---

## Project Structure

| File | Description |
|---|---|
| `my_cnn.py` | CNN for MNIST digit recognition — trains, evaluates, and saves the model |
| `test_my_cnn.py` | Evaluates the saved model on MNIST test set and custom hand-written digits |
| `examine_my_cnn.py` | Visualizes conv1 filters and applies them to a training image via OpenCV |
| `transfer_my_cnn.py` | Transfer learning — adapts the MNIST network to recognize Greek letters (α, β, γ) |
| `NetTransformer-template.py` | Vision Transformer (ViT-style) re-implementation of the digit classifier |
| `experiment.py` | Automated hyperparameter search on Fashion MNIST (~60 experiments) |
| `app.py` | Live webcam digit recognition app built with Gradio |

---

## Setup

```bash
pip install torch torchvision matplotlib opencv-python gradio
```

---

## Usage

### 1. Train the MNIST CNN
```bash
python my_cnn.py
```
Trains for 5 epochs, plots loss and accuracy curves, saves weights to `model.pth`.

### 2. Test on MNIST and custom digits
```bash
python test_my_cnn.py
```
Runs the saved model on the first 10 MNIST test examples and images in `data/my_digits/`.

### 3. Examine conv filters
```bash
python examine_my_cnn.py
```
Prints conv1 weights, plots all 10 filters, and shows the filtered output of the first training image.

### 4. Transfer learning — Greek letters
```bash
python transfer_my_cnn.py
```
Freezes the pretrained MNIST network, replaces the final layer with a 3-node classifier, and trains on images in `data/greek_train/`. Tests on custom images in `data/my_greek/`.

Expected folder structure for Greek data:
```
data/greek_train/
    alpha/
    beta/
    gamma/
data/my_greek/      ← flat folder of test images
```

### 5. Vision Transformer
```bash
python NetTransformer-template.py
```
Trains a patch-based transformer on MNIST for 15 epochs using AdamW.

### 6. Hyperparameter experimentation (Fashion MNIST)
```bash
python experiment.py
```
Runs ~60 experiments sweeping conv filters, dropout, FC size, batch size, and learning rate. Results saved to `experiment_results.csv` and plotted at the end.

### 7. Live digit recognition app
```bash
python app.py
```
Opens a Gradio web app in your browser. Hold a handwritten digit up to your webcam for real-time prediction.

---

## Data

MNIST and Fashion MNIST are downloaded automatically to `./data/` on first run.  
Custom digit images should be placed in `data/my_digits/` (any common image format).
