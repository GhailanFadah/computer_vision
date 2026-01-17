#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>


int greyscale( cv::Mat &src, cv::Mat &dst );
int sepia(cv::Mat &src, cv::Mat &dst);

#endif