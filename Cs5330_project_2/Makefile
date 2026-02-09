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
TARGETS = testing_program run_CBIR

# Default rule
all: $(TARGETS)

# imgDisplay build
testing_program: testing_program.cpp feature_embed.cpp distance_metrics.cpp csv_util.cpp
	$(CXX) $(CXXFLAGS) testing_program.cpp feature_embed.cpp distance_metrics.cpp csv_util.cpp -o test $(OPENCV)

run_CBIR: run_CBIR.cpp feature_embed.cpp distance_metrics.cpp csv_util.cpp
	$(CXX) $(CXXFLAGS) run_CBIR.cpp feature_embed.cpp distance_metrics.cpp csv_util.cpp -o run_CBIR $(OPENCV)

clean:
	rm -f $(TARGETS)