# Real-Time Video Filter Application

A C++ computer vision application that applies various image processing filters to live video streams or static images using OpenCV. Features include edge detection, face detection, depth estimation, and artistic effects.

## 📋 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Filter Controls](#filter-controls)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Author](#author)

## ✨ Features

### Image Processing Filters
- **Grayscale conversion** (built-in and custom implementation)
- **Sepia tone** effect
- **Gaussian blur** (5x5 kernel)
- **Sobel edge detection** (X, Y, and magnitude)
- **Color quantization** with blur
- **Emboss** effect
- **Negative** filter
- **Cartoon** effect
- **Edge glow** effect

### Advanced Features
- **Face detection** with bounding boxes
- **Depth estimation** using neural network (ONNX model)
- **Depth-aware blur** (bokeh effect)
- **Face-targeted effects** (e.g., negative filter on detected faces)
- Real-time processing from webcam or static image input

## 🔧 Prerequisites

- **C++ Compiler** with C++11 support or later (g++, clang++)
- **OpenCV 4.x** or later
- **CMake** 3.10+ (if using CMake build system)
- ONNX Runtime (for depth estimation model)
- Pre-trained depth estimation model: `model_fp16.onnx`

### System Requirements
- Webcam (for live video mode)
- Minimum 4GB RAM
- Modern CPU with SSE2 support

## 📦 Installation

### 1. Install OpenCV

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

#### macOS
```bash
brew install opencv
```

#### Windows
Download pre-built binaries from [OpenCV releases](https://opencv.org/releases/)

### 2. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 3. Compile the Project
```bash
g++ -o vidDisplay vidDisplay.cpp filters.cpp faceDetect.cpp DA2Network.cpp \
    `pkg-config --cflags --libs opencv4`
```

Or using CMake:
```bash
mkdir build && cd build
cmake ..
make
```

### 4. Download Model File
Place the `model_fp16.onnx` depth estimation model in the project directory.

## 🚀 Usage

### Mode 1: Live Video Stream
```bash
./vidDisplay 1
```

### Mode 2: Static Image
```bash
./vidDisplay 2 <path-to-image>
```

**Example:**
```bash
./vidDisplay 2 sample_image.jpg
```

## ⌨️ Filter Controls

Press the following keys while the application is running to apply different filters:

| Key | Filter/Effect | Description |
|-----|---------------|-------------|
| `g` | OpenCV Grayscale | Built-in grayscale conversion |
| `h` | Custom Grayscale | Custom implementation of grayscale |
| `p` | Sepia Tone | Vintage photograph effect |
| `b` | Blur | 5x5 Gaussian blur |
| `x` | Sobel X | Horizontal edge detection |
| `y` | Sobel Y | Vertical edge detection |
| `m` | Magnitude | Combined edge magnitude |
| `l` | Blur Quantize | Color reduction with blur (10 levels) |
| `f` | Face Detection | Detect and draw boxes around faces |
| `d` | Depth Blur | Depth estimation with bokeh effect |
| `e` | Emboss | 3D embossed appearance |
| `n` | Negative | Inverted colors |
| `3` | Face Negative | Negative filter only on detected faces |
| `c` | Cartoon | Cartoon/comic book effect |
| `z` | Edge Glow | Glowing edge effect |
| `s` | Save Frame | Save current filtered frame as `frame.png` |
| `q` | Quit | Exit application |

### Tips
- Press the filter key once to activate it
- Press `s` to save the current view
- Press `q` to quit the application
- Original video displays when no filter is active

## 📁 Project Structure

```
project/
├── vidDisplay.cpp      # Main application file
├── filters.h           # Filter function declarations
├── filters.cpp         # Filter implementations
├── faceDetect.h        # Face detection declarations
├── faceDetect.cpp      # Face detection implementation
├── DA2Network.hpp      # Depth network header
├── DA2Network.cpp      # Depth estimation implementation
├── model_fp16.onnx     # Pre-trained depth model
└── README.md           # This file
```

## 🔬 Technical Details

### Image Processing Techniques
- **Sobel Operators**: 3x3 convolution kernels for edge detection
- **Gaussian Blur**: 5x5 kernel for noise reduction
- **Face Detection**: Haar Cascade or DNN-based detection
- **Depth Estimation**: Deep learning model (ONNX format)

### Performance
- Real-time processing at camera frame rate (typically 30 FPS)
- Depth estimation runs at reduced resolution (0.5x scale factor)
- Face detection with temporal smoothing for stability

### Dependencies
```cpp
#include "opencv2/opencv.hpp"  // Core OpenCV functionality
#include "filters.h"            // Custom filter implementations
#include "faceDetect.h"         // Face detection utilities
#include "DA2Network.hpp"       // Depth estimation network
```

## 🐛 Troubleshooting

### Common Issues

**Camera not opening:**
```
Unable to open video device
```
- Ensure webcam is connected and not in use by another application
- Try changing camera index in `cv::VideoCapture(0)` to `(1)` or `(2)`

**Model file not found:**
- Verify `model_fp16.onnx` is in the same directory as the executable
- Check file permissions

**Compilation errors:**
- Ensure OpenCV is properly installed: `pkg-config --modversion opencv4`
- Verify all source files are present
- Check C++ standard: `-std=c++11` or later

**Slow performance:**
- Depth estimation (`d` filter) is computationally intensive
- Try reducing video resolution or using a faster machine

## 📄 License

This project is created for educational purposes as part of CS5330 coursework.

## 👤 Author

**Ghailan Fadah**  
Northeastern University  
CS5330 - Computer Vision  
January 2026

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the author through the course platform.

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- ONNX Runtime: https://onnxruntime.ai/
- Sobel Operator: https://en.wikipedia.org/wiki/Sobel_operator
