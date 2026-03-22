/*
  Ghailan Fadah
  CS5330
  Harris Corner Detection — live exploration program.

  Hold a printed pattern (chessboard, grid, circles, text) up to the camera.
  Use the trackbars to experiment with the three Harris parameters in real time.

  Controls:
    v — cycle visualisation mode (overlay / response heatmap / side-by-side)
    q — quit
*/
#include <cstdio>
#include <vector>
#include "opencv2/opencv.hpp"

// ── Trackbar state (globals so the callback can reach them) ───────────────────
static int g_blockSize  = 4;    // neighbourhood size  (2–10)
static int g_apertureK  = 1;    // Sobel aperture index → maps to 3,5,7
static int g_kScaled    = 4;    // k * 100  (Harris sensitivity, typically 0.04–0.10)
static int g_threshold  = 150;  // response threshold (0–255 after normalisation)
static int g_visMode    = 0;    // 0=overlay  1=heatmap  2=side-by-side

// Trackbar callbacks — just clamp to legal values
static void onBlockSize (int v, void*) { g_blockSize  = std::max(2, v); }
static void onAperture  (int v, void*) { g_apertureK  = std::max(0, std::min(v, 2)); }
static void onK         (int v, void*) { g_kScaled    = std::max(1, v); }
static void onThreshold (int v, void*) { g_threshold  = v; }

// ── Helpers ───────────────────────────────────────────────────────────────────

// Map trackbar index to Sobel aperture size (must be odd, 3–7)
static int apertureSize() {
    static const int lut[3] = {3, 5, 7};
    return lut[g_apertureK];
}

/*
  Runs Harris, collects corner positions above threshold, and returns
  both the normalised response map and the corner list.
*/
static void runHarris(const cv::Mat &gray,
                      cv::Mat &responseNorm,
                      std::vector<cv::Point> &corners)
{
    cv::Mat response;
    cv::cornerHarris(gray, response,
                     g_blockSize,
                     apertureSize(),
                     g_kScaled / 100.0);

    cv::normalize(response, responseNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

    corners.clear();
    for (int r = 0; r < responseNorm.rows; ++r) {
        const uchar *row = responseNorm.ptr<uchar>(r);
        for (int c = 0; c < responseNorm.cols; ++c) {
            if (row[c] > (uchar)g_threshold)
                corners.emplace_back(c, r);
        }
    }
}

/*
  Draws a small cross at each corner onto `dst`.
  Colour cycles through green → yellow → red as response strength increases.
*/
static void drawCorners(cv::Mat &dst,
                        const std::vector<cv::Point> &corners,
                        const cv::Mat &responseNorm)
{
    for (const auto &pt : corners) {
        uchar strength = responseNorm.at<uchar>(pt);
        // green (weak) → yellow → red (strong)
        int r = std::min(255, (strength - g_threshold) * 4);
        int g = std::max(0,   255 - (strength - g_threshold) * 3);
        cv::drawMarker(dst, pt, cv::Scalar(0, g, r),
                       cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { printf("Cannot open camera\n"); return -1; }

    const std::string WIN = "Harris Corners";
    cv::namedWindow(WIN, cv::WINDOW_AUTOSIZE);

    // Trackbars
    cv::createTrackbar("blockSize (2-10)",  WIN, &g_blockSize, 10,  onBlockSize);
    cv::createTrackbar("aperture (0=3,1=5,2=7)", WIN, &g_apertureK, 2, onAperture);
    cv::createTrackbar("k x100 (1-20)",     WIN, &g_kScaled,   20,  onK);
    cv::createTrackbar("threshold (0-255)", WIN, &g_threshold, 255, onThreshold);

    // Set sensible starting positions
    cv::setTrackbarPos("blockSize (2-10)",  WIN, g_blockSize);
    cv::setTrackbarPos("aperture (0=3,1=5,2=7)", WIN, g_apertureK);
    cv::setTrackbarPos("k x100 (1-20)",     WIN, g_kScaled);
    cv::setTrackbarPos("threshold (0-255)", WIN, g_threshold);

    printf("=== Harris Corner Explorer ===\n");
    printf("  v — cycle view (overlay / heatmap / side-by-side)\n");
    printf("  q — quit\n\n");

    // Colourmap for heatmap view
    const int CMAP = cv::COLORMAP_JET;

    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat responseNorm;
        std::vector<cv::Point> corners;
        runHarris(gray, responseNorm, corners);

        // ── Build display depending on view mode ─────────────────────────────
        cv::Mat display;

        if (g_visMode == 0) {
            // Overlay: draw corners on colour frame
            frame.copyTo(display);
            drawCorners(display, corners, responseNorm);

        } else if (g_visMode == 1) {
            // Heatmap: false-colour response map with corners on top
            cv::applyColorMap(responseNorm, display, CMAP);
            drawCorners(display, corners, responseNorm);

        } else {
            // Side-by-side: original | heatmap, both with corners
            cv::Mat left, right;
            frame.copyTo(left);
            cv::applyColorMap(responseNorm, right, CMAP);
            drawCorners(left,  corners, responseNorm);
            drawCorners(right, corners, responseNorm);
            cv::hconcat(left, right, display);
        }

        // ── HUD ──────────────────────────────────────────────────────────────
        static const char *modeNames[] = {"Overlay", "Heatmap", "Side-by-side"};
        std::string info =
            "Mode: " + std::string(modeNames[g_visMode]) +
            "  |  Corners: " + std::to_string(corners.size()) +
            "  |  blk=" + std::to_string(g_blockSize) +
            " apt=" + std::to_string(apertureSize()) +
            " k=" + std::to_string(g_kScaled / 100) + "." +
            std::to_string(g_kScaled % 100) +
            " thr=" + std::to_string(g_threshold);
        cv::putText(display, info, {8, display.rows - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1, cv::LINE_AA);

        cv::imshow(WIN, display);
        char key = (char)cv::waitKey(10);
        if      (key == 'q') break;
        else if (key == 'v') g_visMode = (g_visMode + 1) % 3;
    }

    return 0;
}