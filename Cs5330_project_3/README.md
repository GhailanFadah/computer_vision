Ghailan Fadah 
24/2/2026
*using one travel day for project*
built and ran on MacOS Tahoe 26.2 using visual studio code as the IDE
# 2D Object Detection and Classification

A real-time 2D object detection system built in C++ with OpenCV that segments objects from camera input or static images and classifies them using either hand-computed features or CNN embeddings.

## Overview

The system converts input to a binary image, cleans it with morphological operations, and performs connected component analysis to segment objects. It supports two operational modes:

- **Training mode** — Label detected objects and store their features in a database
- **Classification mode** — Identify objects by comparing against stored features

Two feature extraction methods are available:

- **Hand-computed** — 9-dimensional feature vector (percent filled, aspect ratio, 7 Hu moments) with scaled Euclidean distance
- **CNN** — 512-dimensional embedding from ResNet18's penultimate layer with cosine distance

## Requirements

- C++17 or later
- OpenCV 4.x (with `dnn` module)
- ResNet18 ONNX model file (`resnet18-v2-7.onnx`) — required for CNN mode

## Building

From the project root directory:

```bash
cd computer_vision
make
```

## Usage

```bash
# Camera input with hand-computed features
./detect 1 hand

# Camera input with CNN embeddings
./detect 1 cnn

# Static image with hand-computed features
./detect 2 hand path/to/image.jpg

# Static image with CNN embeddings
./detect 2 cnn path/to/image.jpg
```

## Controls

| Key | Action |
|-----|--------|
| `t` | Enter training mode — label the current object |
| `n` | Label an unknown object (when confidence is low) |
| `q` | Quit |

## Project Structure

```
computer_vision/
├── objectDetect.cpp    # Main application loop, input handling, display
├── utilObject.cpp      # Image processing, feature computation, classification
├── utilObject.h        # Struct definitions and function declarations
├── utilities.cpp       # CNN embedding extraction and ROI preparation
├── Makefile            # Build configuration
├── resnet18-v2-7.onnx  # ResNet18 model (required for CNN mode)
├── object_db.csv       # Hand-computed feature database (generated at runtime)
├── embedding_db.csv    # CNN embedding database (generated at runtime)
└── readme/
    └── README.md       # This file
```

## Pipeline

1. **Preprocessing** — 5×5 Gaussian blur to reduce noise
2. **Thresholding** — Otsu's method for adaptive binarization
3. **Morphological cleaning** — Opening and closing with a 7×7 elliptical kernel
4. **Segmentation** — Connected component analysis to identify regions
5. **Feature extraction** — Either hand-computed moments or CNN embedding
6. **Classification** — Nearest-neighbor matching against the stored database

## Feature Details

### Hand-Computed (2 features)

- **Percent filled** — Ratio of region area to oriented bounding box area
- **Aspect ratio** — Ratio of longer to shorter side of the bounding box

### CNN Embedding (512 features)

The oriented region is extracted, axis-aligned, resized to 224×224, and passed through ResNet18. The output is taken from the flatten layer before the final classification head, producing a 512-dimensional feature vector.

## Databases

Feature databases are CSV files generated automatically during training. If switching feature methods or updating the feature vector format, delete the corresponding database file and retrain:

```bash
rm object_db.csv      # Reset hand-computed database
rm embedding_db.csv   # Reset CNN embedding database
```

## Author

Ghailan Fadah — CS5330, Spring 2026