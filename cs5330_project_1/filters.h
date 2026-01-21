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

#endif