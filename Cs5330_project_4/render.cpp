/*
  Ghailan Fadah
  CS5330
  Program 2 — AR Rendering
  
  Loads camera_matrix.csv + distortion.csv, detects the chessboard live,
  and overlays:
    1. 3D coordinate axes (X=red, Y=green, Z=blue)
    2. A wireframe cube sitting on the board
    3. A wireframe pyramid sitting on the board
    4. A wireframe + filled face floating above the board

  Controls:
    1  — toggle axes
    2  — toggle cube
    3  — toggle pyramid
    4  - toggle spacecraft
    q  — quit
*/
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "Utils.h"

// ── Drawing helpers ───────────────────────────────────────────────────────────

/*
  Projects a list of 3-D world points to 2-D image points given the
  current pose (rvec / tvec) and camera intrinsics.
*/
static std::vector<cv::Point2f> projectPts(
    const std::vector<cv::Point3f> &pts3d,
    const cv::Mat &rvec, const cv::Mat &tvec,
    const cv::Mat &cam,  const cv::Mat &dist)
{
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts3d, rvec, tvec, cam, dist, pts2d);
    return pts2d;
}

/*
  Fills a polygon face and then draws its wireframe outline on top.
  `idx` is the list of vertex indices into the projected point array `p`.
*/
static void drawFace(cv::Mat &img,
                     const std::vector<cv::Point2f> &p,
                     const std::vector<int> &idx,
                     cv::Scalar fillCol,
                     cv::Scalar lineCol,
                     double alpha = 0.55)
{
    std::vector<cv::Point> pts;
    for (int i : idx)
        pts.emplace_back((int)p[i].x, (int)p[i].y);

    // Filled face blended with the background for a translucent look
    cv::Mat overlay;
    img.copyTo(overlay);
    cv::fillConvexPoly(overlay, pts, fillCol, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img);

    // Wireframe outline on top
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i)
        cv::line(img, pts[i], pts[(i+1)%n], lineCol, 1, cv::LINE_AA);
}

/*
  Draws a spacecraft floating above the board centre.

  Geometry (all Z values negative = above the board):
    - Fuselage  : tapered hexagonal prism along X axis
    - Cockpit   : small box on top of the fuselage nose
    - Wings     : two swept quads extending from mid-fuselage
    - Engines   : two small rectangular boxes at the wing tips
    - Tail fin  : vertical quad at the rear

  The craft is centred at the board's mid-point and hovers at Z = -hover.
*/
static void drawSpacecraft(cv::Mat &img,
                           const cv::Mat &rvec, const cv::Mat &tvec,
                           const cv::Mat &cam,  const cv::Mat &dist)
{
    // Board centre in world units (4×4 squares from origin for a 9×6 board)
    const float cx = 4.0f, cy = 2.5f;
    const float hover = 3.5f;   // height above board

    // ── Fuselage: tapered box, long axis along X ─────────────────────────────
    // Front (nose) is narrower than the rear
    const float fL = 3.5f;  // half-length
    const float fWf = 0.4f, fHf = 0.3f;  // front half-width / half-height
    const float fWr = 0.7f, fHr = 0.55f; // rear  half-width / half-height

    //  0–3 : front face  (x = cx+fL)
    //  4–7 : rear  face  (x = cx-fL)
    std::vector<cv::Point3f> fus = {
        {cx+fL, cy-fWf, -hover-fHf}, {cx+fL, cy+fWf, -hover-fHf},
        {cx+fL, cy+fWf, -hover+fHf}, {cx+fL, cy-fWf, -hover+fHf},
        {cx-fL, cy-fWr, -hover-fHr}, {cx-fL, cy+fWr, -hover-fHr},
        {cx-fL, cy+fWr, -hover+fHr}, {cx-fL, cy-fWr, -hover+fHr},
    };
    auto pf = projectPts(fus, rvec, tvec, cam, dist);
    cv::Scalar fuseFill(40, 40, 120), fuseWire(100, 180, 255);
    drawFace(img, pf, {0,1,2,3}, fuseFill, fuseWire);        // nose
    drawFace(img, pf, {4,5,6,7}, fuseFill, fuseWire);        // tail
    drawFace(img, pf, {0,4,7,3}, fuseFill, fuseWire);        // left side
    drawFace(img, pf, {1,5,6,2}, fuseFill, fuseWire);        // right side
    drawFace(img, pf, {3,7,6,2}, fuseFill, fuseWire);        // top
    drawFace(img, pf, {0,4,5,1}, fuseFill, fuseWire);        // bottom

    // ── Cockpit: small box sitting on top of the nose half ───────────────────
    const float ckx = cx + 1.8f;
    std::vector<cv::Point3f> ck = {
        {ckx+0.5f, cy-0.25f, -hover+fHf},
        {ckx+0.5f, cy+0.25f, -hover+fHf},
        {ckx-0.5f, cy+0.25f, -hover+fHf},
        {ckx-0.5f, cy-0.25f, -hover+fHf},
        {ckx+0.3f, cy-0.2f,  -hover+fHf+0.45f},
        {ckx+0.3f, cy+0.2f,  -hover+fHf+0.45f},
        {ckx-0.5f, cy+0.2f,  -hover+fHf+0.45f},
        {ckx-0.5f, cy-0.2f,  -hover+fHf+0.45f},
    };
    auto pc = projectPts(ck, rvec, tvec, cam, dist);
    cv::Scalar ckFill(180, 220, 255), ckWire(220, 240, 255);
    drawFace(img, pc, {0,1,2,3}, ckFill, ckWire);
    drawFace(img, pc, {4,5,6,7}, ckFill, ckWire);
    drawFace(img, pc, {0,4,7,3}, ckFill, ckWire);
    drawFace(img, pc, {1,5,6,2}, ckFill, ckWire);
    drawFace(img, pc, {4,5,1,0}, ckFill, ckWire);
    drawFace(img, pc, {7,6,2,3}, ckFill, ckWire);

    // ── Wings: two swept quads (port and starboard) ───────────────────────────
    // Each wing root attaches at mid-fuselage sides, tip sweeps back and out
    const float wx = cx - 0.3f;  // root X
    const float wr = fWr;        // root Y extent (matches fuselage rear width)
    const float wSpan = 2.8f, wSweep = 1.6f, wThick = 0.12f;

    // Starboard wing (positive Y)
    std::vector<cv::Point3f> ws = {
        {wx,        cy+wr,          -hover},           // root front bottom
        {wx-wSweep, cy+wr+wSpan,    -hover},           // tip  front bottom
        {wx-wSweep, cy+wr+wSpan,    -hover-wThick},    // tip  front top
        {wx,        cy+wr,          -hover-wThick},    // root front top
        {wx-fL*0.9f,cy+wr,          -hover},           // root rear  bottom
        {wx-wSweep-0.4f,cy+wr+wSpan,-hover},           // tip  rear  bottom
        {wx-wSweep-0.4f,cy+wr+wSpan,-hover-wThick},    // tip  rear  top
        {wx-fL*0.9f,cy+wr,          -hover-wThick},    // root rear  top
    };
    // Port wing (negative Y, mirror)
    std::vector<cv::Point3f> wp;
    for (auto &v : ws)
        wp.push_back({v.x, 2*cy - v.y, v.z});

    cv::Scalar wingFill(30, 60, 140), wingWire(80, 160, 255);
    for (auto &wing : {ws, wp}) {
        auto pw = projectPts(wing, rvec, tvec, cam, dist);
        drawFace(img, pw, {0,1,2,3},     wingFill, wingWire); // front edge
        drawFace(img, pw, {4,5,6,7},     wingFill, wingWire); // rear edge
        drawFace(img, pw, {3,2,6,7},     wingFill, wingWire); // top surface
        drawFace(img, pw, {0,1,5,4},     wingFill, wingWire); // bottom surface
        drawFace(img, pw, {1,2,6,5},     wingFill, wingWire); // tip
    }

    // ── Engine nacelles: small boxes at wing tips ─────────────────────────────
    auto drawEngine = [&](float ex, float ey) {
        float ew=0.55f, eh=0.22f, ed=0.22f;
        std::vector<cv::Point3f> eng = {
            {ex+ew, ey-ed, -hover-eh}, {ex+ew, ey+ed, -hover-eh},
            {ex+ew, ey+ed, -hover+eh}, {ex+ew, ey-ed, -hover+eh},
            {ex-ew, ey-ed, -hover-eh}, {ex-ew, ey+ed, -hover-eh},
            {ex-ew, ey+ed, -hover+eh}, {ex-ew, ey-ed, -hover+eh},
        };
        auto pe = projectPts(eng, rvec, tvec, cam, dist);
        cv::Scalar engFill(0, 100, 180), engWire(0, 220, 255);
        drawFace(img, pe, {0,1,2,3}, engFill, engWire);
        drawFace(img, pe, {4,5,6,7}, engFill, engWire);
        drawFace(img, pe, {0,4,7,3}, engFill, engWire);
        drawFace(img, pe, {1,5,6,2}, engFill, engWire);
        drawFace(img, pe, {3,7,6,2}, engFill, engWire);
        drawFace(img, pe, {0,4,5,1}, engFill, engWire);
        // Engine glow ring
        cv::Point2f glow = projectPts({{ex-ew, ey, -hover}},
                                       rvec, tvec, cam, dist)[0];
        cv::circle(img, glow, 5, {0, 255, 180}, -1, cv::LINE_AA);
        cv::circle(img, glow, 8, {0, 200, 140},  1, cv::LINE_AA);
    };

    float tipY_s = cy + wr + wSpan;
    float tipY_p = cy - wr - wSpan;
    float tipX   = wx - wSweep - 0.2f;
    drawEngine(tipX, tipY_s);
    drawEngine(tipX, tipY_p);

    // ── Tail fin: vertical quad at the rear ───────────────────────────────────
    std::vector<cv::Point3f> fin = {
        {cx-fL,       cy, -hover+fHr},
        {cx-fL*0.55f, cy, -hover+fHr},
        {cx-fL*0.55f, cy, -hover+fHr+1.1f},
        {cx-fL,       cy, -hover+fHr+0.5f},
    };
    auto pfin = projectPts(fin, rvec, tvec, cam, dist);
    cv::Scalar finFill(60, 40, 160), finWire(140, 100, 255);
    drawFace(img, pfin, {0,1,2,3}, finFill, finWire);
}

// Draw XYZ axes rooted at the board origin.
static void drawAxes(cv::Mat &img,
                     const cv::Mat &rvec, const cv::Mat &tvec,
                     const cv::Mat &cam,  const cv::Mat &dist,
                     float len = 3.0f)
{
    std::vector<cv::Point3f> pts = {
        {0,0,0}, {len,0,0}, {0,len,0}, {0,0,-len}  // -Z points "up" out of board
    };
    auto p = projectPts(pts, rvec, tvec, cam, dist);
    cv::arrowedLine(img, p[0], p[1], {0,0,255},   2, cv::LINE_AA, 0, 0.2); // X red
    cv::arrowedLine(img, p[0], p[2], {0,255,0},   2, cv::LINE_AA, 0, 0.2); // Y green
    cv::arrowedLine(img, p[0], p[3], {255,100,0}, 2, cv::LINE_AA, 0, 0.2); // Z blue

    cv::putText(img, "X", p[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255},   2);
    cv::putText(img, "Y", p[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0},   2);
    cv::putText(img, "Z", p[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,100,0}, 2);
}

// Draw a wireframe cube.  Base sits on Z=0, top at Z=-size (above board).
static void drawCube(cv::Mat &img,
                     const cv::Mat &rvec, const cv::Mat &tvec,
                     const cv::Mat &cam,  const cv::Mat &dist,
                     float ox = 1.0f, float oy = 1.0f, float size = 2.0f)
{
    float s = size;
    // 8 corners: bottom ring (z=0) then top ring (z=-s)
    std::vector<cv::Point3f> c3d = {
        {ox,   oy,    0}, {ox+s, oy,    0}, {ox+s, oy+s,  0}, {ox,   oy+s,  0},  // bottom
        {ox,   oy,   -s}, {ox+s, oy,   -s}, {ox+s, oy+s, -s}, {ox,   oy+s, -s}   // top
    };
    auto p = projectPts(c3d, rvec, tvec, cam, dist);
    cv::Scalar col(0, 220, 255); int t = 2;

    // Bottom face
    for (int i = 0; i < 4; ++i) cv::line(img, p[i], p[(i+1)%4], col, t, cv::LINE_AA);
    // Top face
    for (int i = 0; i < 4; ++i) cv::line(img, p[i+4], p[(i+1)%4+4], col, t, cv::LINE_AA);
    // Vertical edges
    for (int i = 0; i < 4; ++i) cv::line(img, p[i], p[i+4], col, t, cv::LINE_AA);
}

// Draw a wireframe square-base pyramid.
static void drawPyramid(cv::Mat &img,
                        const cv::Mat &rvec, const cv::Mat &tvec,
                        const cv::Mat &cam,  const cv::Mat &dist,
                        float ox = 5.0f, float oy = 1.0f, float size = 2.0f)
{
    float s = size;
    float h = s * 1.5f;  // height
    std::vector<cv::Point3f> p3d = {
        {ox,   oy,    0}, {ox+s, oy,    0},  // base corners
        {ox+s, oy+s,  0}, {ox,   oy+s,  0},
        {ox+s/2, oy+s/2, -h}                 // apex
    };
    auto p = projectPts(p3d, rvec, tvec, cam, dist);
    cv::Scalar col(0, 165, 255); int t = 2;

    // Base square
    for (int i = 0; i < 4; ++i) cv::line(img, p[i], p[(i+1)%4], col, t, cv::LINE_AA);
    // Lateral edges to apex
    for (int i = 0; i < 4; ++i) cv::line(img, p[i], p[4], col, t, cv::LINE_AA);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    // Load calibration
    cv::Mat camMat   = loadMatCSV("camera_matrix.csv");
    cv::Mat distCoef = loadMatCSV("distortion.csv");

    if (camMat.empty() || distCoef.empty()) {
        printf("Error: could not load calibration CSVs.\n"
               "Run the calibrate program first.\n");
        return -1;
    }
    printf("Loaded calibration data.\n");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error: cannot open camera\n");
        return -1;
    }

    cv::namedWindow("AR Render", cv::WINDOW_AUTOSIZE);

    std::vector<cv::Point3f> objPts = buildObjectPoints();

    bool showAxes       = true;
    bool showCube       = true;
    bool showPyramid    = true;
    bool showSpacecraft = true;
    bool printPose      = false;

    printf("=== AR Render ===\n");
    printf("  1 — toggle axes\n  2 — toggle cube\n  3 — toggle pyramid\n");
    printf("  4 — toggle spacecraft\n  p — toggle pose printing\n  q — quit\n\n");

    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::vector<cv::Point2f> corners;
        bool found = findBoardCorners(frame, corners, /*draw=*/false);

        if (found) {
            cv::Mat rvec, tvec;
            cv::solvePnP(objPts, corners, camMat, distCoef, rvec, tvec);

            if (showAxes)       drawAxes      (frame, rvec, tvec, camMat, distCoef);
            if (showCube)       drawCube      (frame, rvec, tvec, camMat, distCoef);
            if (showPyramid)    drawPyramid   (frame, rvec, tvec, camMat, distCoef);
            if (showSpacecraft) drawSpacecraft(frame, rvec, tvec, camMat, distCoef);

            cv::putText(frame, "Board detected", {10,30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);

            if (printPose) {
                printf("R: [%7.3f, %7.3f, %7.3f]  "
                       "T: [%7.3f, %7.3f, %7.3f]\n",
                       rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                       tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            }
        } else {
            cv::putText(frame, "Board NOT found", {10,30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,255}, 2);
        }

        // Overlay toggle state
        auto toggleTxt = [](bool on) { return on ? "[ON] " : "[OFF]"; };
        cv::putText(frame, "1:Axes "    + std::string(toggleTxt(showAxes)),
                    {10,60},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);
        cv::putText(frame, "2:Cube "    + std::string(toggleTxt(showCube)),
                    {10,82},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);
        cv::putText(frame, "3:Pyramid " + std::string(toggleTxt(showPyramid)),
                    {10,104}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);
        cv::putText(frame, "4:Spacecraft " + std::string(toggleTxt(showSpacecraft)),
                    {10,126}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);

        cv::putText(frame, "p:Pose "       + std::string(toggleTxt(printPose)),
                    {10,148}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);
        cv::putText(frame, "q:Quit ", {10,170}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);

        cv::imshow("AR Render", frame);
        char key = (char)cv::waitKey(10);

        if      (key == 'q') break;
        else if (key == '1') showAxes       = !showAxes;
        else if (key == '2') showCube       = !showCube;
        else if (key == '3') showPyramid    = !showPyramid;
        else if (key == '4') showSpacecraft = !showSpacecraft;
        else if (key == 'p') {
            printPose = !printPose;
            printf("Pose printing %s\n", printPose ? "ON" : "OFF");
        }
    }

    return 0;
}