/*
Ghailan Fadah
1/14/26
CS5330
Program reads in a live stream from camera or a still image and allows applying different filters
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "filters.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
int main(int argc, char *argv[]) {
    /*
    function streams a video from device camera and allows different filters and processing to the stream
    input: N/A
    output <int> 0
    */

        int mode = std::stoi(argv[1]);

        if(mode == 2 && argc < 3 || mode > 2 && mode < 1){
        printf("usage: %s <mode 1 or 2> <image filename>\n ", argv[0]);
        exit(-1);
        }
        
        // make a DANetwork object
        DA2Network da_net( "model_fp16.onnx" );
        const float reduction = 0.5;

        cv::VideoCapture *capdev;
        capdev = new cv::VideoCapture(0);
        
        char filename[256];
        

        // open the video device
        if(mode == 1){
            //cv::VideoCapture *capdev;
            //capdev = new cv::VideoCapture(0);
            if( !capdev->isOpened() ) {
                    printf("Unable to open video device\n");
                    return(-1);
            }
            // get some properties of the image
            cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
             printf("Expected size: %d %d\n", refS.width, refS.height);

           
        }else if(mode == 2){
            //char filename[256];
            strcpy(filename, argv[2]);
        }
        float scale_factor = 1.0;

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
        cv::Mat my_depth;
        cv::Mat depth_vis;
        cv::Mat my_emboss;
        cv::Mat my_negative;
        cv::Mat my_cartoon;
        cv::Mat my_glow;
        std::vector<cv::Rect> faces;
        cv::Rect last(0, 0, 0, 0);

       
        char dis;

        for(;;) {
                if(mode == 2){
                    frame = cv::imread(filename);
                }else if (mode == 1){
                    *capdev >> frame; // get a new frame from the camera, treat as a stream
                }
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
                }else if (dis == 'd'){
                    cv::resize( frame, frame, cv::Size(), reduction, reduction );
                    da_net.set_input( frame, scale_factor );
                    da_net.run_network( my_depth, frame.size() );
                    blurDepth(frame, depth_vis, my_depth);
                    cv::imshow("Video", depth_vis);
                }else if(dis == 'e'){
                    sobelY3x3(frame, raw_my_Ysobel);
                    sobelX3x3(frame, raw_my_Xsobel);
                    emboss(raw_my_Xsobel, raw_my_Ysobel, my_emboss);
                    cv::imshow("Video", my_emboss);
                }else if (dis == 'n') {
                    negative(frame, my_negative);
                    cv::imshow("Video", my_negative);
                }else if (dis == '3'){
                    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);
                    detectFaces( grey, faces );
                    drawBoxes( frame, faces );
                    for (const auto &face : faces) {
                        cv::Mat roi_src = frame(face);
                        cv::Mat roi_dst;

                        negative(roi_src, roi_dst);
                        roi_dst.copyTo(frame(face));
                    }
                    cv::imshow("Video", frame);
                }else if (dis == 'c'){
                    cartoon(frame, my_cartoon);
                    cv::imshow("Video", my_cartoon);
                }else if (dis == 'z'){
                    edgeGlow(frame, my_glow);
                    cv::imshow("Video", my_glow);
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
                }else if(key == 'd'){
                    dis = 'd';
                }else if(key == 'e'){
                    dis = 'e';
                }else if(key == 'n'){
                    dis = 'n';
                }else if(key == '3'){
                    dis = '3';
                }else if(key == 'c'){
                    dis = 'c';
                }else if(key == 'z'){
                    dis = 'z';
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
                    }else if(dis == 'd'){
                        imwrite("frame.png", depth_vis);
                    }else if(dis == 'e'){
                        imwrite("frame.png", my_emboss);
                    }else if(dis == 'n'){
                        imwrite("frame.png", my_negative);
                    }else if(dis == '3'){
                        imwrite("frame.png", frame);
                    }else if(dis == 'c'){
                        imwrite("frame.png", my_cartoon);
                    }else if(dis == 'z'){
                        imwrite("frame.png", my_glow);
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