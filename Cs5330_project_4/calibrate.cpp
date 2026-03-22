/*
  Ghailan Fadah
  CS5330
  Program 1 — Camera Calibration
  
  Controls:
    s  — save current frame's corners as a calibration sample
    c  — run calibration with collected samples (need >= 5)
    q  — quit
*/
#include <cstdio>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Utils.h"

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error: cannot open camera\n");
        return -1;
    }

    cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);

    // Accumulated calibration data across saved frames
    std::vector<std::vector<cv::Point2f>>  allCorners;   // 2-D image points
    std::vector<std::vector<cv::Point3f>>  allObjPts;    // 3-D world points

    std::vector<cv::Point3f> objPtsTemplate = buildObjectPoints();

    bool calibrated = false;
    cv::Mat camMat, distCoeffs;

    printf("=== Calibration Program ===\n");
    printf("  s — save frame\n  c — calibrate (need >= 5 frames)\n  q — quit\n\n");

    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display;
        frame.copyTo(display);

        std::vector<cv::Point2f> corners;
        bool found = findBoardCorners(display, corners, /*draw=*/true);

        // HUD
        std::string status = found ? "Board found" : "Board NOT found";
        cv::Scalar col = found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        cv::putText(display, status,      {10, 30},  cv::FONT_HERSHEY_SIMPLEX, 0.8, col, 2);
        cv::putText(display, "Saved: " + std::to_string(allCorners.size()),
                    {10, 60},  cv::FONT_HERSHEY_SIMPLEX, 0.7, {255,200,0}, 2);
        if (calibrated) {
            cv::putText(display, "CALIBRATED — saved to CSV",
                        {10, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,255}, 2);
        }

        cv::imshow("Calibration", display);
        char key = (char)cv::waitKey(10);

        if (key == 'q') break;

        // ── Save frame ───────────────────────────────────────────────────────
        if (key == 's') {
            if (!found) {
                printf("No board found — move the board into view first.\n");
            } else {
                allCorners.push_back(corners);
                allObjPts.push_back(objPtsTemplate);
                printf("Saved frame %zu\n", allCorners.size());
            }
        }

        // ── Run calibration ──────────────────────────────────────────────────
        if (key == 'c') {
            if ((int)allCorners.size() < 5) {
                printf("Need at least 5 saved frames (have %zu).\n", allCorners.size());
                continue;
            }

            cv::Size imgSize(frame.cols, frame.rows);

            // Build the initial camera matrix guess from image size.
            // fx = fy ≈ image width (reasonable starting focal length);
            // principal point at image centre.
         
            cv::Mat camMat0 = (cv::Mat_<double>(3,3) <<
                1.0, 0,   imgSize.width  / 2.0,
                0,   1.0, imgSize.height / 2.0,
                0,   0,   1.0);
            cv::Mat dist0 = cv::Mat::zeros(1, 5, CV_64F);

            printf("\n=== Before Calibration (initial guess) ===\n");
            printf("Camera matrix:\n");
            for (int r = 0; r < 3; ++r) {
                printf("  [ ");
                for (int c = 0; c < 3; ++c)
                    printf("%12.4f ", camMat0.at<double>(r,c));
                printf("]\n");
            }
            printf("Distortion coefficients:\n  [ ");
            for (int c = 0; c < dist0.cols; ++c)
                printf("%10.6f ", dist0.at<double>(0,c));
            printf("]\n\n");

            // Run calibration (camMat / distCoeffs are output parameters)
            std::vector<cv::Mat> rvecs, tvecs;
            double rpe = cv::calibrateCamera(
                allObjPts, allCorners, imgSize,
                camMat, distCoeffs,
                rvecs, tvecs,
                cv::CALIB_FIX_ASPECT_RATIO);

            printf("=== After Calibration ===\n");
            printf("Camera matrix:\n");
            for (int r = 0; r < 3; ++r) {
                printf("  [ ");
                for (int c = 0; c < 3; ++c)
                    printf("%12.4f ", camMat.at<double>(r,c));
                printf("]\n");
            }
            printf("Distortion coefficients:\n  [ ");
            for (int c = 0; c < distCoeffs.cols; ++c)
                printf("%10.6f ", distCoeffs.at<double>(0,c));
            printf("]\n\n");
            printf("Re-projection error: %.4f px\n\n", rpe);

            // Save to CSV
            if (saveMatCSV("camera_matrix.csv", camMat) &&
                saveMatCSV("distortion.csv",    distCoeffs))
            {
                printf("Saved camera_matrix.csv and distortion.csv\n");
                calibrated = true;
            } else {
                printf("Error saving CSV files.\n");
            }
        }
    }

    return 0;
}