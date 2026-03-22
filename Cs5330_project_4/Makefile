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
TARGETS = calibrate render harris

# Default rule
all: $(TARGETS)

# imgDisplay build
calibrate: calibrate.cpp Utils.cpp
	$(CXX) $(CXXFLAGS) calibrate.cpp Utils.cpp -o cali $(OPENCV)

render: render.cpp Utils.cpp
	$(CXX) $(CXXFLAGS) render.cpp Utils.cpp -o render $(OPENCV)

harris: harris.cpp 
	$(CXX) $(CXXFLAGS) harris.cpp -o har $(OPENCV)

clean:
	rm -f $(TARGETS)