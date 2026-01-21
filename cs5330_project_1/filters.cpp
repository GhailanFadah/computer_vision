/*
Ghailan Fadah
1/14/26
CS5330
Program creates different filters for image processing
*/

#include "opencv2/opencv.hpp"
#include <cmath>

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
   [1,2,4,2,1
    2,4,8,4,2
    4,8,16,8,4
    2,4,8,4,2
    1,2,4,2,1]
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
    [1,2,4,2,1
    2,4,8,4,2
    4,8,16,8,4
    2,4,8,4,2
    1,2,4,2,1]
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

int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
     /*
    function create a custom sobel 3X3 filter in x-direction using pointers to access pxs
    [-1,0,1
    -2,0,2
    -1,0,1]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i<src.rows-1; i++){
        cv::Vec3b *s_pt_up = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *s_pt_down = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *d_pt = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j<src.cols-1; j++){

            for(int c=0;c<3;c++){

                int res = s_pt_up[j-1][c]*-1 + s_pt_up[j][c]*0 + s_pt_up[j+1][c]*1
                + s_pt_mid[j-1][c]*-2 + s_pt_mid[j][c]*0 + s_pt_mid[j+1][c]*2
                + s_pt_down[j-1][c]*-1 + s_pt_down[j][c]*0 + s_pt_down[j+1][c]*1;

                // range [-255-255]
                d_pt[j][c] = res;

            }
        }
    }
    return(0);
}

int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
     /*
    function create a custom sobel 3X3 filter in y-direction using pointers to access pxs
    [1,2,1
    0,0,0
    -1,-2,-1]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i<src.rows-1; i++){
        cv::Vec3b *s_pt_up = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *s_pt_down = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *d_pt = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j<src.cols-1; j++){

            for(int c=0;c<3;c++){

                int res = s_pt_up[j-1][c]*1 + s_pt_up[j][c]*2 + s_pt_up[j+1][c]*1
                + s_pt_mid[j-1][c]*0 + s_pt_mid[j][c]*0 + s_pt_mid[j+1][c]*0
                + s_pt_down[j-1][c]*-1 + s_pt_down[j][c]*-2 + s_pt_down[j+1][c]*-1;

                // range [-255-255]
                d_pt[j][c] = res;

            }
        }
    }

    return (0);

}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
      /*
    function determines mag of sx, sy and claps mag val to [0-255] range
    
    input: cv::Mat &sx, cv::Mat &sy, cv::Mat &dst
    output <int> 0
    */


    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i<sx.rows; i++){
        cv::Vec3s *px = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *py = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<sx.cols; j++){

             for(int c=0;c<3;c++){

                int mag = std::sqrt(px[j][c]*px[j][c] + py[j][c]*py[j][c]);
                pd[j][c] = cv::saturate_cast<uchar>(mag);
             }

        }
    }

    return(0);

}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels){
       /*
    function blurs and quants input based on number of levels
    
    input: cv::Mat &src, cv::Mat &dst, int levels 
    output <int> 0
    */

    cv::Mat tem;

    blur5x5_2(src, tem);
    tem.copyTo(dst);

    int bucket = 256/levels;

    for (int i = 0; i<src.rows; i++){
        cv::Vec3b *ps = tem.ptr<cv::Vec3b>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j<src.cols; j++){

             for(int c=0;c<3;c++){

                int res = ps[j][c]/bucket;
                pd[j][c] = res*bucket;
             }

        }
    }

    return(0);
}

int blurDepth(cv::Mat &src, cv::Mat &dst, cv::Mat &depth){
       /*
    function blurs image based on Depth: further = more blur
    
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(31, 31), 0);
    //blur5x5_2(src, blur);
    src.copyTo(dst);

    

    for (int i = 0; i<src.rows; i++){
        uchar* dptr = depth.ptr<uchar>(i);
        cv::Vec3b *ps = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *pb = blur.ptr<cv::Vec3b>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j<src.cols; j++){

            float dep = dptr[j] / 255.0; 
            float wt = 1.0 - dep;

             for(int c=0;c<3;c++){

                pd[j][c] = static_cast<uchar>((1 - wt) * ps[j][c] + wt * pb[j][c]);
             }

        }
    }

    return(0);
}

int emboss( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
      /*
    function uses sobel X,Y to add embossing effect
    
    input: cv::Mat &sx, cv::Mat &sy, cv::Mat &dst
    output <int> 0
    */


    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i<sx.rows; i++){
        cv::Vec3s *px = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *py = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<sx.cols; j++){

             for(int c=0;c<3;c++){

                int em = px[j][c] * 0.7071 + py[j][c] * 0.7071;
                pd[j][c] = cv::saturate_cast<uchar>(em * 2 + 128);
             }

        }
    }

    return(0);

}

int negative(cv::Mat &src, cv::Mat &dst){
      /*
    function takes the negative of an image
    
    input: cv::Mat &sx, cv::Mat &sy, cv::Mat &dst
    output <int> 0
    */


    src.copyTo(dst);

    for (int i = 0; i<src.rows; i++){
        cv::Vec3b *ps = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<src.cols; j++){

             for(int c=0;c<3;c++){

                pd[j][c] = 255 - ps[j][c];
             }

        }
    }

    return(0);

}