/*
Ghailan Fadah
2/23/26
CS5330
Headers for my object detection program
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct RegionFeatures {
    double area;
    double percentFilled;
    double aspectRatio;
    cv::Point2d centroid;
    double orientation;   // radians
    cv::RotatedRect obb;  // oriented bounding box
};

struct FeatureVector
{
    double percentFilled;
    double aspectRatio;
};

struct AxisExtents {
    float minE1, maxE1;  // extent along primary axis
    float minE2, maxE2;  // extent along secondary axis
};

struct DBEntry
{
    std::string label;
    FeatureVector features;
};

struct EmbeddingEntry
{
    std::string label;
    cv::Mat embedding;  // 1 x 512 float
};


int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int threshold(cv::Mat &src, cv::Mat &dst);
int clean_binary(cv::Mat &src, cv::Mat &dst);
int findLargestRegion(const cv::Mat& stats);
cv::Mat displayRegionMap(const cv::Mat& labels, const cv::Mat& stats, int minArea);
RegionFeatures computeRegionFeatures(const cv::Mat& labels, int regionID);
void drawPrimaryAxis(cv::Mat& img, const RegionFeatures& f);
void drawOBB(cv::Mat& img, const RegionFeatures& f);
void saveFeatureToFile(const std::string& filename, const std::string& label, const FeatureVector& v);
std::vector<DBEntry> loadDatabase(const std::string& filename);
FeatureVector extractFeatureVector(const RegionFeatures& f);
FeatureVector computeStdDev(const std::vector<DBEntry>& db);
double computeDistance(const FeatureVector& a, const FeatureVector& b, const FeatureVector& stddev);
std::string classifyObject(const FeatureVector& unknown, const std::vector<DBEntry>& db, const FeatureVector& stddev);

AxisExtents computeAxisExtents(const cv::Mat& labels, int regionID, const RegionFeatures& f);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,float minE1, float maxE1, float minE2, float maxE2, int debug);
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void saveEmbeddingToFile(const std::string& filename, const std::string& label, const cv::Mat& embedding);
std::vector<EmbeddingEntry> loadEmbeddingDatabase(const std::string& filename);
double computeEmbeddingDistance(const cv::Mat& a, const cv::Mat& b);
std::string classifyByEmbedding(const cv::Mat& unknown, const std::vector<EmbeddingEntry>& db);

#endif