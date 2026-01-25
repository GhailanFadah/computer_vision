/*
Ghailan Fadah
1/14/26
CS5330
Headers for my filters program
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>


int greyscale( cv::Mat &src, cv::Mat &dst );
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int blurDepth(cv::Mat &src, cv::Mat &dst, cv::Mat &depth);
int emboss( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int negative(cv::Mat &src, cv::Mat &dst);
int cartoon(cv::Mat &src, cv::Mat &dst);
int thresholdEdges(cv::Mat &src, cv::Mat &dst, int thresh);
int strongBlur(cv::Mat &src, cv::Mat &dst, int iter);
int addGlow(cv::Mat &src, cv::Mat &glow, cv::Mat &dst, float strength);
int edgeGlow(cv::Mat &src, cv::Mat &dst);

#endif