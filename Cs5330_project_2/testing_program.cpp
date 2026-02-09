/*
Ghailan Fadah
1/28/26
CS5330
Testing program between 2 images instead of whole dataset
*/

#include "feature_embed.h"
#include "distance_metrics.h"
#include "csv_util.h"
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

int test_cnn(){
    char csv[] = "ResNet18_olym.csv";

    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;

    read_image_data_csv(csv, filenames, data, 1);

    return (0);
}

int main(int argc, char *argv[]) {

    cv::Mat src;
    cv::Mat src2;
    cv::Mat res;
    cv::Mat res2;
    char filename[256];

    test_cnn();

    //ensure filename is given 
    if(argc < 2){
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }
    strcpy(filename, argv[1]);

    // read img
    src = cv::imread(filename);
    src2 = cv::imread("data/sunset2.jpg");

    // check if file is able to be opened
    if(src.data == NULL){
        printf("Unable to read image%s\n", filename);
        exit(-1);
    }

    sunset_matching(src, res);
    sunset_matching(src2, res2);
   
    float distance = Inter_Dis(res,res2);

    printf("distnace%f\n", distance);


}