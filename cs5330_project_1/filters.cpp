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

int blur5x5_1(cv::Mat &src, cv::Mat &dst){
    /*
    function create a custom gauss 5X5 filter using at to access pxs
    12421
    24842
    481684
    24842
    12421
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */
    src.copyTo(dst);

    for (int i = 2; i<src.rows-2; i++){
        for(int j = 2; j<src.cols-2; j++){
            for(int c=0; c<3; c++){

                // access the 5X5 and convs
                int res = src.at<cv::Vec3b>(i-2, j-2)[c] + src.at<cv::Vec3b>(i-2, j-1)[c]*2 + src.at<cv::Vec3b>(i-2, j)[c]*4 + src.at<cv::Vec3b>(i-2, j+1)[c]*2 + src.at<cv::Vec3b>(i-2, j+2)[c]
                + src.at<cv::Vec3b>(i-1, j-2)[c]*2 + src.at<cv::Vec3b>(i-1, j-1)[c]*4 + src.at<cv::Vec3b>(i-1, j)[c]*8 + src.at<cv::Vec3b>(i-1, j+1)[c]*4 + src.at<cv::Vec3b>(i-1, j+2)[c]*2
                + src.at<cv::Vec3b>(i, j-2)[c]*4 + src.at<cv::Vec3b>(i, j-1)[c]*8 + src.at<cv::Vec3b>(i, j)[c]*16 + src.at<cv::Vec3b>(i, j+1)[c]*8 + src.at<cv::Vec3b>(i, j+2)[c]*4
                + src.at<cv::Vec3b>(i+1, j-2)[c]*2 + src.at<cv::Vec3b>(i+1, j-1)[c]*4 + src.at<cv::Vec3b>(i+1, j)[c]*8 + src.at<cv::Vec3b>(i+1, j+1)[c]*4 + src.at<cv::Vec3b>(i+1, j+2)[c]*2
                + src.at<cv::Vec3b>(i+2, j-2)[c] + src.at<cv::Vec3b>(i+2, j-1)[c]*2 + src.at<cv::Vec3b>(i+2, j)[c]*4 + src.at<cv::Vec3b>(i+2, j+1)[c]*2 + src.at<cv::Vec3b>(i+2, j+2)[c];

                // scales range back to [0,255]
                dst.at<cv::Vec3b>(i,j)[c] = res/100;

            }
        }
    }

    return(0);

}

int blur5x5_2(cv::Mat &src, cv::Mat &dst){
    /*
    function create a custom gauss 5X5 filter using pointers to access pxs
    [12421
    24842
    481684
    24842
    12421]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    cv::Mat tmp;
    src.copyTo(tmp);

    // loop conv using [1,2,4,2,1] filter
    for(int i=0;i<src.rows;i++){
        cv:: Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv:: Vec3b *t_pt = tmp.ptr<cv::Vec3b>(i);
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){

                int res = s_pt_mid[j-2][c] + s_pt_mid[j-1][c]*2 + s_pt_mid[j][c]*4 + s_pt_mid[j+1][c]*2 + s_pt_mid[j+2][c];

                // scales range back to [0,255]
                t_pt[j][c]= res/10;

            }
        }

    }

    tmp.copyTo(dst);

    // loop conv using [1,2,4,2,1] filter
    for(int i=2;i<src.rows-2;i++){
        // pointers for each row and dst row
        cv:: Vec3b *s_pt_up2 = tmp.ptr<cv::Vec3b>(i-2);
        cv:: Vec3b *s_pt_up1 = tmp.ptr<cv::Vec3b>(i-1);
        cv:: Vec3b *s_pt_mid = tmp.ptr<cv::Vec3b>(i);
        cv:: Vec3b *s_pt_down1 = tmp.ptr<cv::Vec3b>(i+1);
        cv:: Vec3b *s_pt_down2 = tmp.ptr<cv::Vec3b>(i+2);
        cv:: Vec3b *d_pt = dst.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){

                int res = s_pt_up2[j][c] + s_pt_up1[j][c]*2 + s_pt_mid[j][c]*4 + s_pt_down1[j][c]*2 + s_pt_down2[j][c];

                // scales range back to [0,255]
                d_pt[j][c] = res/10;

            }
        }

    }

    return(0);
}