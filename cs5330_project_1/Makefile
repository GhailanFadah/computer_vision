# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall

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
	$(CXX) $(CXXFLAGS) vidDisplay.cpp filters.cpp faceDetect.cpp -o vidDisplay $(OPENCV)

timeBlur: vidDisplay.cpp filters.cpp
	$(CXX) $(CXXFLAGS) timeBlur.cpp filters.cpp -o timeBlur $(OPENCV)

showFace: showFaces.cpp faceDetect.cpp
	$(CXX) $(CXXFLAGS) faceDetect.cpp showFaces.cpp -o showFace $(OPENCV)

# Clean
clean:
	rm -f $(TARGETS)
