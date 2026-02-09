/*
Ghailan Fadah
1/28/26
CS5330
Headers for my implementing different distance metrics
*/
#include <opencv2/opencv.hpp>

float SSD(cv::Mat &A,cv::Mat &B);
float Inter_Dis(cv::Mat &A,cv::Mat &B);
float cosine_dis(const std::vector<float> &A,const std::vector<float> &B);