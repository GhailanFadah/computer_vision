/*
Ghailan Fadah
2/23/26
CS5330
Program performs object detection on a given input. allows the user to add to its database. 
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "utilObject.h"

// Forward declare utility function from utilities.cpp
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                        float minE1, float maxE1, float minE2, float maxE2, int debug);

int main(int argc, char *argv[]) {

    int mode = std::stoi(argv[1]);
    std::string featureMode = argv[2]; 

    if ((mode == 2 && argc < 4) || (mode < 1 || mode > 2)) {
        printf("usage: %s <1=camera|2=image> <hand|cnn> [image filename]\n", argv[0]);
        return -1;
    }

    if (featureMode != "hand" && featureMode != "cnn") {
        printf("Feature mode must be 'hand' or 'cnn'\n");
        return -1;
    }


    cv::VideoCapture capdev;
    char filename[256];

    if (mode == 1) {
        capdev.open(0);
        if (!capdev.isOpened()) {
            printf("Unable to open video device\n");
            return -1;
        }
        cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                       (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);
    } else if (mode == 2) {
        strcpy(filename, argv[3]);
    }

    cv::namedWindow("Video", 1);

    cv::Mat frame, gray, binary, clean_img;
    cv::Mat labels, stats, centroids;
    cv::Mat img_obb, RegionMap;

    // Hand-computed feature database
    std::vector<DBEntry> database = loadDatabase("object_db.csv");
    FeatureVector stddev = computeStdDev(database);

    // cnn embedding database + network
    cv::dnn::Net net;
    std::vector<EmbeddingEntry> embDatabase;

    if (featureMode == "cnn") {
        net = cv::dnn::readNet("resnet18-v2-7.onnx");
        if (net.empty()) {
            printf("Error: could not load ResNet18 ONNX model\n");
            return -1;
        }
        embDatabase = loadEmbeddingDatabase("embedding_db.csv");
        printf("Loaded %zu embedding entries\n", embDatabase.size());
    }

    RegionFeatures feat;

    for (;;) {

        if (mode == 2) {
            frame = cv::imread(filename);
        } else if (mode == 1) {
            capdev >> frame;
        }

        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cv::Mat original;
        frame.copyTo(original);  

        blur5x5_2(frame, frame);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        threshold(gray, binary);
        clean_binary(binary, clean_img);

        int numLabels = cv::connectedComponentsWithStats(clean_img, labels, stats, centroids, 8, CV_32S);
        int maxLabel = findLargestRegion(stats);

        // testing purpose
        //cv::imshow("Grayscale", gray);
        //cv::imshow("Binary", binary);
        //cv::imshow("Cleaned", clean_img);

        if (maxLabel > 0) {
            RegionMap = displayRegionMap(labels, stats, 500);
            feat = computeRegionFeatures(labels, maxLabel);
            cv::cvtColor(clean_img, img_obb, cv::COLOR_GRAY2BGR);
            drawPrimaryAxis(img_obb, feat);
            drawOBB(frame, feat);

            std::string arText = "AR: " + std::to_string(feat.aspectRatio).substr(0, 5);
            cv::putText(img_obb, arText, cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(255, 0, 0), 2);

            
            std::string classLabel = "Unknown";

            if (featureMode == "hand" && !database.empty()) {
                FeatureVector unknown = extractFeatureVector(feat);
                classLabel = classifyObject(unknown, database, stddev);

            } else if (featureMode == "cnn") {
                // Compute axis extents for ROI extraction
                AxisExtents ext = computeAxisExtents(labels, maxLabel, feat);

                
                cv::Mat embImage;
                prepEmbeddingImage(original, embImage,
                                   (int)feat.centroid.x, (int)feat.centroid.y,
                                   (float)feat.orientation,
                                   ext.minE1, ext.maxE1,
                                   ext.minE2, ext.maxE2, 0);

                if (!embImage.empty()) {
                    
                    cv::Mat embInput;
                    if (embImage.channels() == 1)
                        cv::cvtColor(embImage, embInput, cv::COLOR_GRAY2BGR);
                    else
                        embInput = embImage;

                    // Compute embedding
                    cv::Mat embedding;
                    getEmbedding(embInput, embedding, net, 0);

                    // Classify if we have entries
                    if (!embDatabase.empty()) {
                        classLabel = classifyByEmbedding(embedding, embDatabase);
                    }
                }
            }

            if (classLabel == "Unknown"){
                cv::putText(frame, "Unknown object. Press 'n' to label this object",
                            cv::Point(20, 120),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 0, 255), 2);
            }else{
            cv::putText(frame, "Label: " + classLabel,
                        cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(255, 0, 0), 2);
            }

            // testing
            //cv::imshow("Region Map", RegionMap);
            //cv::imshow("Features", img_obb);

            cv::imshow("Detection", frame);
        }

        char key = cv::waitKey(10);

        if (key == 'q') {
            break;

        } else if (key == 't' || key == 'n') {
            if (maxLabel > 0) {
                std::string label;
                std::cout << "Enter object label: ";
                std::cin >> label;

                if (featureMode == "hand") {
                    FeatureVector fv = extractFeatureVector(feat);
                    saveFeatureToFile("object_db.csv", label, fv);

                    database = loadDatabase("object_db.csv");
                    stddev = computeStdDev(database);

                } else if (featureMode == "cnn") {
                    AxisExtents ext = computeAxisExtents(labels, maxLabel, feat);

                    cv::Mat embImage;
                    prepEmbeddingImage(original, embImage,
                                       (int)feat.centroid.x, (int)feat.centroid.y,
                                       (float)feat.orientation,
                                       ext.minE1, ext.maxE1,
                                       ext.minE2, ext.maxE2, 0);

                    if (!embImage.empty()) {
                        cv::Mat embInput;
                        if (embImage.channels() == 1)
                            cv::cvtColor(embImage, embInput, cv::COLOR_GRAY2BGR);
                        else
                            embInput = embImage;

                        cv::Mat embedding;
                        getEmbedding(embInput, embedding, net, 0);

                        saveEmbeddingToFile("embedding_db.csv", label, embedding);
                        embDatabase = loadEmbeddingDatabase("embedding_db.csv");
                    }
                }

                std::cout << "Training sample saved.\n";
            }
        }
    }

    return 0;
}