# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall

# OpenCV flags
OPENCV = `pkg-config --cflags --libs opencv4`

# Targets
TARGETS = imgDisplay vidDisplay timeBlur

# Default rule
all: $(TARGETS)

# imgDisplay build
imgDisplay: imgDisplay.cpp
	$(CXX) $(CXXFLAGS) imgDisplay.cpp -o imgDisplay $(OPENCV)

# vidDisplay build
vidDisplay: vidDisplay.cpp filters.cpp
	$(CXX) $(CXXFLAGS) vidDisplay.cpp filters.cpp -o vidDisplay $(OPENCV)

timeBlur: vidDisplay.cpp filters.cpp
	$(CXX) $(CXXFLAGS) timeBlur.cpp filters.cpp -o timeBlur $(OPENCV)

# Clean
clean:
	rm -f $(TARGETS)
