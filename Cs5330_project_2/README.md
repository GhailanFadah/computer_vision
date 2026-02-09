# 🖼️ Content-Based Image Retrieval (CBIR) System

A C++ and Python-based image retrieval system that finds similar images using computer vision techniques including color histograms, texture analysis, and deep neural network features. Features both a command-line interface and a lightweight GUI built with Tkinter.

**Course**: CS5330 - Computer Vision  
**Author**: Ghailan Fadah  
**Date**: January 2026

---

## 📋 Overview

This CBIR system analyzes images using various feature extraction methods and distance metrics to find visually similar images in a dataset. The system supports multiple feature representations and matching strategies, making it versatile for different image retrieval tasks.

### Key Capabilities
- **6 feature extraction methods**: Baseline, color histograms, multi-region histograms, texture, sunset detection, and DNN embeddings
- **2 distance metrics**: Sum of Squared Differences (SSD) and Histogram Intersection
- **Python GUI**: Interactive interface for easy image selection and result visualization
- **Efficient matching**: Priority queue-based top-N retrieval
- **Flexible architecture**: Function pointers for swappable feature/distance strategies

---

## 🎯 Key Functionalities

### Feature Extraction Methods

1. **Baseline** (`baseline`)
   - Extracts 7×7 center region of image
   - Simple pixel-level comparison
   - Fast but limited accuracy

2. **RGB Histogram** (`hist`)
   - 16×16 normalized rg chromaticity histogram
   - Color-invariant to intensity changes
   - Good for color-based retrieval

3. **Multi-Region Histogram** (`2hist`)
   - Separate 8×8 histograms for top and bottom halves
   - Captures spatial color distribution
   - Useful for images with distinct regions (e.g., sky/ground)

4. **Histogram + Texture** (`hist-text`)
   - Combines 16×16 color histogram with texture features
   - Texture computed using Sobel edge magnitude
   - 512-dimensional feature vector
   - Best for images with both color and texture characteristics

5. **Sunset Detection** (`sun`)
   - Custom warmth-saturation 2D histogram
   - Focuses on sky region (top half)
   - Optimized for sunset/sunrise images

6. **Deep Neural Network** (`dnn`)
   - Uses pre-computed embeddings from CSV
   - Cosine distance matching
   - Highest accuracy for general image retrieval

### Distance Metrics

- **SSD (Sum of Squared Differences)**: `(A[i] - B[i])²` summed across all features
- **Histogram Intersection**: `1 - Σ min(A[i], B[i])` - optimal for histogram matching

---

## 🔧 Installation

### Prerequisites
- **C++ Compiler**: g++ with C++11 support
- **OpenCV 4.x**: Computer vision library
- **Python 3.8+**: For GUI
- **Tkinter**: Python GUI library
- **Pillow**: Python imaging library

### Build Instructions

1. **Compile the C++ executable**:
   ```bash
   g++ -o run_CBIR run_CBIR.cpp feature_embed.cpp distance_metrics.cpp \
       csv_util.cpp `pkg-config --cflags --libs opencv4` -std=c++11
   ```

2. **Verify Python dependencies**:
   ```bash
   pip install pillow
   # Tkinter is usually included with Python
   ```

3. **Prepare dataset**:
   - Place images in `./data` directory
   - Supported formats: `.jpg`, `.png`, `.ppm`, `.tif`

---

## 🚀 Usage

### Command Line Interface

```bash
./run_CBIR <target_image> <data_directory> <feature_type> <matching_type> <N>
```

**Parameters**:
- `target_image`: Path to query image
- `data_directory`: Directory containing image dataset
- `feature_type`: `baseline`, `hist`, `2hist`, `hist-text`, `sun`, or `dnn`
- `matching_type`: `SSD` or `i` (intersection)
- `N`: Number of top matches to return

**Example**:
```bash
./run_CBIR ./query.jpg ./data hist i 5
```

**Output**:
```
./data/sunset1.jpg,0.023
./data/sunset2.jpg,0.045
./data/sunset3.jpg,0.067
...
```

### Python GUI

```bash
python GUI.py
```

**Workflow**:
1. Click **"Select Image"** to choose target image
2. Set parameters:
   - **Top N**: Number of results (default: 5)
   - **Feature**: Choose feature extraction method
   - **Matching**: Choose distance metric
3. Click **"Run CBIR"** to execute search
4. View results with thumbnails and distances

**GUI Features**:
- Visual target image preview
- Scrollable results panel
- Thumbnail previews of matches
- Distance scores for each match

---

## 📁 Project Structure

```
.
├── run_CBIR.cpp              # Main retrieval program
├── feature_embed.cpp         # Feature extraction implementations
├── feature_embed.h           # Feature function declarations
├── distance_metrics.cpp      # Distance metric implementations
├── distance_metrics.h        # Distance function declarations
├── csv_util.cpp/.h           # CSV reading utilities (for DNN features)
├── GUI.py                    # Python Tkinter interface
└── data/                     # Image dataset directory
```

---

## 🔬 Technical Details

### Feature Vector Dimensions

| Feature Type | Dimensions | Data Type |
|--------------|-----------|-----------|
| Baseline | 7×7×3 = 147 | CV_32FC3 |
| RGB Histogram | 16×16 = 256 | CV_32FC1 |
| Multi-Region | 128 | CV_32FC1 |
| Hist+Texture | 512 | CV_32FC1 |
| Sunset | 256 | CV_32FC1 |
| DNN | Varies | float vector |

### Algorithm Flow

1. **Load target image** and extract features
2. **Iterate through dataset**:
   - Load each image
   - Extract features using selected method
   - Compute distance to target features
   - Maintain top-N matches using max-heap
3. **Output results** sorted by distance (ascending)

### Optimization Techniques

- **Priority Queue**: O(log N) insertion for top-N tracking
- **Histogram Normalization**: Invariant to image size
- **Efficient Pointer Access**: Direct memory access for pixel operations
- **Function Pointers**: Runtime strategy selection without virtual calls

---

## 📊 Performance Considerations

**Feature Extraction Speed** (approximate, 640×480 image):
- Baseline: <1ms
- Histogram: ~5ms
- Texture: ~15ms (includes Sobel computation)
- DNN: Depends on pre-computation

**Memory Usage**:
- Feature vectors: <1KB per image
- Priority queue: O(N) where N = top results

---

## 🎨 Usage Examples

### Finding Similar Landscapes
```bash
./run_CBIR landscape.jpg ./data 2hist i 10
```
Uses multi-region histogram to match sky and ground separately.

### Texture-Based Search
```bash
./run_CBIR fabric.jpg ./data hist-text SSD 5
```
Combines color and texture for fabric/pattern matching.

### Sunset Image Collection
```bash
./run_CBIR sunset.jpg ./data sun i 8
```
Custom feature optimized for warm-colored sky images.

---

## 🐛 Troubleshooting

**Issue**: "Cannot open directory"
- **Solution**: Verify `./data` directory exists and contains images

**Issue**: GUI images not displaying
- **Solution**: Ensure Pillow is installed: `pip install pillow`

**Issue**: Compilation errors
- **Solution**: Check OpenCV installation: `pkg-config --modversion opencv4`

**Issue**: Python module not found
- **Solution**: Install Tkinter: `sudo apt-get install python3-tk` (Linux)

---

## 🔮 Future Enhancements

- [ ] Add more distance metrics (Chi-square, Bhattacharyya)
- [ ] Implement relevance feedback
- [ ] GPU acceleration for feature extraction
- [ ] Web-based interface with Flask
- [ ] Support for video frame retrieval
- [ ] Automatic feature selection based on query type

---

## 📚 References

- **Color Histograms**: Swain & Ballard (1991) "Color Indexing"
- **Histogram Intersection**: Superior for normalized histograms
- **rg Chromaticity**: Intensity-invariant color representation
- **Sobel Operators**: Edge-based texture features

---

## 📄 License

Educational project for CS5330 coursework at Northeastern University.

---

**Built with C++, OpenCV, and Python**