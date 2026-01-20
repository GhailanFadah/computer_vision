/*
Ghailan Fadah
1/14/26
CS5330
Program reads in a live stream from camera and allows applying different filters
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "filters.h"
#include "faceDetect.h"
int main(int argc, char *argv[]) {
    /*
    function streams a video from device camera and allows different filters and processing to the stream
    input: N/A
    output <int> 0
    */
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        cv::Mat grey;
        cv::Mat my_gray;
        cv::Mat my_sepia;
        cv::Mat my_blur;
        cv::Mat raw_my_Xsobel;
        cv::Mat raw_my_Ysobel;
        cv::Mat my_Xsobel;
        cv::Mat my_Ysobel;
        cv::Mat my_mag;
        cv::Mat my_quant;
        std::vector<cv::Rect> faces;
        cv::Rect last(0, 0, 0, 0);

       
        char dis = 'c';

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }  
                
                if (dis == 'g'){
                    cvtColor(frame, grey, cv::COLOR_BGRA2GRAY, 0);
                    cv::imshow("Video", grey);
                }else if (dis == 'h'){
                    greyscale(frame, my_gray);
                    cv::imshow("video", my_gray);
                }else if ( dis == 'p'){
                    sepia(frame, my_sepia);
                    cv::imshow("Video", my_sepia);
                }else if (dis == 'b') {
                    blur5x5_2(frame, my_blur);
                    cv::imshow("Video", my_blur);
                }else if(dis == 'x'){
                    sobelX3x3(frame, raw_my_Xsobel);
                    cv::convertScaleAbs(raw_my_Xsobel, my_Xsobel);
                    cv::imshow("Video", my_Xsobel);
                }else if (dis == 'y'){
                    sobelY3x3(frame, raw_my_Ysobel);
                    cv::convertScaleAbs(raw_my_Ysobel, my_Ysobel);
                    cv::imshow("Video", my_Ysobel);
                }else if(dis == 'm'){
                    sobelY3x3(frame, raw_my_Ysobel);
                    sobelX3x3(frame, raw_my_Xsobel);
                    magnitude(raw_my_Xsobel, raw_my_Ysobel, my_mag);
                    cv::imshow("Video", my_mag);
                }else if(dis == 'l'){
                    blurQuantize(frame, my_quant, 10);
                    cv::imshow("Video", my_quant);
                }else if(dis == 'f'){
                    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);
                    detectFaces( grey, faces );
                    drawBoxes( frame, faces );

                    if( faces.size() > 0 ) {
                        last.x = (faces[0].x + last.x)/2;
                        last.y = (faces[0].y + last.y)/2;
                        last.width = (faces[0].width + last.width)/2;
                        last.height = (faces[0].height + last.height)/2;
                    }

                    cv::imshow("Video", frame);
                }else{
                    cv::imshow("Video", frame);
                }
                

                // see if there is a waiting keystroke
                char key = cv::waitKey(10);

                if (key == 'g'){
                    dis = 'g';
                }else if (key == 'h'){
                    dis = 'h';
                }else if (key == 'p'){
                    dis = 'p';
                }else if(key == 'b'){
                    dis = 'b';
                }else if(key == 'x'){
                    dis = 'x';
                }else if(key == 'y'){
                    dis = 'y';
                }else if(key == 'm'){
                    dis = 'm';
                }else if(key == 'l'){
                    dis = 'l';
                }else if(key == 'f'){
                    dis = 'f';
                }else if (key == 's'){

                    if (dis == 'g'){
                        imwrite("frame.png", grey);
                    }else if(dis == 'h'){
                    imwrite("frame.png", my_gray);
                    }else if (dis == 'p'){
                        imwrite("frame.png", my_sepia);
                    }else if(dis == 'b'){
                        imwrite("frame.png", my_blur);
                    }else if (dis == 'x'){
                        imwrite("frame.png", my_Xsobel);
                    }else if (dis == 'y'){
                        imwrite("frame.png", my_Ysobel);
                    }else if(dis == 'm'){
                        imwrite("frame.png", my_mag);
                    }else if(dis == 'l'){
                        imwrite("frame.png", my_quant);
                    }else if(dis == 'f'){
                        imwrite("frame.png", frame);
                    }else{
                        imwrite("frame.png", frame);
                    }
                }else if( key == 'q'){
                    break;
                }
               
        }

        delete capdev;
        return(0);
}