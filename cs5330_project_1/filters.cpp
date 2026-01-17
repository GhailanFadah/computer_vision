/*
Ghailan Fadah
1/14/26
CS5330
Program creates different filters for image processing
*/

#include "opencv2/opencv.hpp"

int greyscale(cv::Mat &src, cv::Mat &dst){
    /*
    function create a custom grayscale filter
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    
    
    dst.create(src.size(), src.type());

    for (int i = 0; i<src.rows; i++){
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *ptr2 = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<src.cols; j++){

            // set each color channel to the value 255 - B
            char blue_value = ptr[j][0];
            ptr2[j][2] = 255 - blue_value;
            ptr2[j][1] = 255 - blue_value;
            ptr2[j][0] = 255 - blue_value;
        }
    }

    

    return 0;

}

int sepia(cv::Mat &src, cv::Mat &dst){
    /*
    function create a custom sepia filter
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */


    
    dst.create(src.size(), src.type());

    for (int i = 0; i<src.rows; i++){
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *ptr2 = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<src.cols; j++){
            
            // gets values for each color channel
            char blue_value = ptr[j][0];
            char green_value = ptr[j][1];
            char red_value = ptr[j][2];

            // new values
            int new_b = 0.131*blue_value + 0.534*green_value + 0.272*red_value;
            int new_g = 0.168*blue_value + 0.686*green_value + 0.349*red_value;
            int new_r = 0.189*blue_value + 0.769*green_value + 0.393*red_value;

            // applies weight for each color channel if less than 255 else assign 255
            ptr2[j][0] = (new_b > 255) ? 255:new_b;
            ptr2[j][1] = (new_g > 255) ? 255:new_g;
            ptr2[j][2] = (new_r > 255) ? 255:new_r;
        }
    }

    

    return 0;

}