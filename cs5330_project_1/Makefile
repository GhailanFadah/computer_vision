# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall

ONNX_INCLUDE = /Users/ghailanfadah/Desktop/onnxruntime-osx-arm64-1.16.3/include
ONNX_LIB     = /Users/ghailanfadah/Desktop/onnxruntime-osx-arm64-1.16.3/lib
ONNX_LDFLAGS = -L$(ONNX_LIB) -lonnxruntime -Wl,-rpath,$(ONNX_LIB)

# OpenCV flags
OPENCV = `pkg-config --cflags --libs opencv4`

# Targets
TARGETS = imgDisplay vidDisplay timeBlur showFace

# Default rule
all: $(TARGETS)

# imgDisplay build
imgDisplay: imgDisplay.cpp
	$(CXX) $(CXXFLAGS) imgDisplay.cpp -o imgDisplay $(OPENCV)

# vidDisplay build
vidDisplay: vidDisplay.cpp filters.cpp faceDetect.cpp
	$(CXX) $(CXXFLAGS) -I$(ONNX_INCLUDE) vidDisplay.cpp filters.cpp faceDetect.cpp -o vidDisplay $(OPENCV) $(ONNX_LDFLAGS)

timeBlur: vidDisplay.cpp filters.cpp
	$(CXX) $(CXXFLAGS) timeBlur.cpp filters.cpp -o timeBlur $(OPENCV)

showFace: showFaces.cpp faceDetect.cpp
	$(CXX) $(CXXFLAGS) faceDetect.cpp showFaces.cpp -o showFace $(OPENCV)

# Clean
clean:
	rm -f $(TARGETS)
