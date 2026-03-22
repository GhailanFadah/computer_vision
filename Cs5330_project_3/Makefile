# Ghailan Fadah
# 1/28/26
# CS5330
# Makefile for project 2

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall
# OpenCV flags
OPENCV = `pkg-config --cflags --libs opencv4`

# Targets
TARGETS = objectDetect

# Default rule
all: $(TARGETS)

# imgDisplay build
objectDetect: objectDetect.cpp utilObject.cpp
	$(CXX) $(CXXFLAGS) utilities.cpp objectDetect.cpp utilObject.cpp -o vid $(OPENCV)

clean:
	rm -f $(TARGETS)