/*
Ghailan Fadah
1/28/26
CS5330
Header file for different feature embeddings
*/

#include <opencv2/opencv.hpp>

int baseline_matching(cv::Mat &src, cv::Mat &res);
int hist_matching(cv::Mat &src, cv::Mat &res);
int hist_bottom_matching(cv::Mat &src, cv::Mat &res);
int hist_bottom_matching(cv::Mat &src, cv::Mat &res);
int hist_top_bottom_matching(cv::Mat &src, cv::Mat &res);
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int hist_texture_rg_matching(cv::Mat &src, cv::Mat &res);
int run_csv_matching(const char* target_image, const char* csv_file,int N);
struct MatchResult {std::string filename;float dis;};
int sunset_matching(cv::Mat &src, cv::Mat &res);