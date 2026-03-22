/*
  Ghailan Fadah
  CS5330
  Implementation of shared calibration/AR utilities.
*/
#include "Utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

// ── Corner detection ─────────────────────────────────────────────────────────

bool findBoardCorners(cv::Mat &frame,
                      std::vector<cv::Point2f> &corners,
                      bool draw)
{
    cv::Size boardSize(BOARD_COLS, BOARD_ROWS);

    bool found = cv::findChessboardCorners(
        frame, boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // Refine to sub-pixel accuracy
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(
            gray, corners,
            cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                             30, 0.001));

        if (draw)
            cv::drawChessboardCorners(frame, boardSize, corners, found);
    }

    return found;
}

std::vector<cv::Point3f> buildObjectPoints()
{
    std::vector<cv::Point3f> pts;
    pts.reserve(BOARD_COLS * BOARD_ROWS);
    for (int r = 0; r < BOARD_ROWS; ++r)
        for (int c = 0; c < BOARD_COLS; ++c)
            pts.emplace_back(c * SQUARE_SIZE, r * SQUARE_SIZE, 0.0f);
    return pts;
}

// ── CSV I/O ──────────────────────────────────────────────────────────────────

bool saveMatCSV(const std::string &filename, const cv::Mat &m)
{
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "saveMatCSV: cannot open " << filename << "\n";
        return false;
    }
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            if (c) f << ',';
            f << m.at<double>(r, c);
        }
        f << '\n';
    }
    return true;
}

cv::Mat loadMatCSV(const std::string &filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "loadMatCSV: cannot open " << filename << "\n";
        return {};
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> row;
        while (std::getline(ss, tok, ','))
            row.push_back(std::stod(tok));
        rows.push_back(row);
    }

    if (rows.empty()) return {};
    int nrows = (int)rows.size();
    int ncols = (int)rows[0].size();

    cv::Mat m(nrows, ncols, CV_64F);
    for (int r = 0; r < nrows; ++r)
        for (int c = 0; c < ncols; ++c)
            m.at<double>(r, c) = rows[r][c];

    return m;
}