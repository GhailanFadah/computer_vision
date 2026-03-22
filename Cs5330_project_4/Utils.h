/*
  Ghailan Fadah
  CS5330
  Shared utilities for camera calibration and AR rendering.
*/
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

// ── Board configuration ──────────────────────────────────────────────────────
constexpr int   BOARD_COLS    = 8;   // inner corners along width
constexpr int   BOARD_ROWS    = 6;   // inner corners along height
constexpr float SQUARE_SIZE   = 1.0f; // world units per square (set to mm if you prefer)

// ── Corner detection ─────────────────────────────────────────────────────────

/*
  Tries to find and refine chessboard corners in `frame`.
  Returns true and fills `corners` if the full board is found.
  Draws the corners onto `frame` in-place when `draw` is true.
*/
bool findBoardCorners(cv::Mat &frame,
                      std::vector<cv::Point2f> &corners,
                      bool draw = true);

/*
  Builds the vector of ideal 3-D object points for one chessboard view.
  Points lie on the Z=0 plane, spaced SQUARE_SIZE apart.
*/
std::vector<cv::Point3f> buildObjectPoints();

// ── CSV I/O ──────────────────────────────────────────────────────────────────

/*
  Saves a cv::Mat (any size, CV_64F) to a plain CSV file.
  Each matrix row becomes one line; values are comma-separated.
*/
bool saveMatCSV(const std::string &filename, const cv::Mat &m);

/*
  Loads a cv::Mat from a CSV file written by saveMatCSV.
  Returns an empty Mat on failure.
*/
cv::Mat loadMatCSV(const std::string &filename);

#endif // CALIB_UTILS_H