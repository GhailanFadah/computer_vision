/*
Ghailan Fadah
1/14/26
CS5330
Program reads in an image and displays it using cv
*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

int main(int argc, char *argv[]){
    /*
    function takes a file name from the command line and displays it in a window
    input: <str> filename
    output <int> 0
    */
    cv::Mat src;
    char filename[256];

    //ensure filename is given 
    if(argc < 2){
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }
    strcpy(filename, argv[1]);

    // read img
    src = cv::imread(filename);

    // check if file is able to be opened
    if(src.data == NULL){
        printf("Unable to read image%s\n", filename);
        exit(-1);
    }

    // creates window and shows img
    imshow(filename, src);

    // keeps window up till q is pressed
    while (true){
    char k = cv::waitKey(5);

        if (k == 'q'){
            cv::destroyWindow(filename);
            break;
        }

    }

    return 0;

}





