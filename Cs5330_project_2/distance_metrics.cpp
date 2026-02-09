/*
Ghailan Fadah
1/28/26
CS5330
Program creates different distance metrics to determine distance between vectors
*/

#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstdlib>

float SSD(cv::Mat &A,cv::Mat &B) {
    float res = 0.0f;

   

    for (int i = 0; i < A.rows; i++) {
        const cv::Vec3f *pa = A.ptr<cv::Vec3f>(i);
        const cv::Vec3f *pb = B.ptr<cv::Vec3f>(i);

        for (int j = 0; j < A.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float diff = pa[j][c] - pb[j][c];
                res += diff * diff;
            }
        }
    }
    return res;
}

float Inter_Dis(cv::Mat &A,cv::Mat &B){
    float sum = 0.0;
    for (int i = 0; i < A.rows; i++){
        const float *pa = A.ptr<float>(i);
        const float *pb = B.ptr<float>(i);

        for (int j = 0; j < A.cols; j++){

            sum += std::min(pa[j], pb[j]);
        }
    }

    return 1 - sum; 
}

float cosine_dis(const std::vector<float> &A,const std::vector<float> &B){


    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (size_t i = 0; i < A.size(); i++) {
        dot   += A[i] * B[i];
        normA += A[i] * A[i];
        normB += B[i] * B[i];
    }

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    return 1.0f - (dot / (normA * normB));
}


