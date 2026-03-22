/*
Ghailan Fadah
2/23/26
CS5330
Program defines certain function needed for 2d object detection. 
*/

#include <fstream>
#include <cmath>
#include <sstream>
#include "utilObject.h"


int blur5x5_2(cv::Mat &src, cv::Mat &dst){
     /*
    function create a custom gauss 5X5 filter using pointers to access pxs
    [1,2,4,2,1
    2,4,8,4,2
    4,8,16,8,4
    2,4,8,4,2
    1,2,4,2,1]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */
    cv::Mat tmp;
    src.copyTo(tmp);

    for(int i=0;i<src.rows;i++){
        cv::Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *t_pt = tmp.ptr<cv::Vec3b>(i);
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                int res = s_pt_mid[j-2][c] + s_pt_mid[j-1][c]*2 + s_pt_mid[j][c]*4 + s_pt_mid[j+1][c]*2 + s_pt_mid[j+2][c];
                t_pt[j][c]= res/10;
            }
        }
    }

    tmp.copyTo(dst);

    for(int i=2;i<src.rows-2;i++){
        cv::Vec3b *s_pt_up2 = tmp.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *s_pt_up1 = tmp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s_pt_mid = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b *s_pt_down1 = tmp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *s_pt_down2 = tmp.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *d_pt = dst.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                int res = s_pt_up2[j][c] + s_pt_up1[j][c]*2 + s_pt_mid[j][c]*4 + s_pt_down1[j][c]*2 + s_pt_down2[j][c];
                d_pt[j][c] = res/10;
            }
        }
    }

    return(0);
}

int threshold(cv::Mat &src, cv::Mat &dst){

     /*Compute Otsu threshold: finds the value that minimizes
     intra-class variance between foreground and background
    */

    dst.create(src.size(), CV_8UC1);

    double otsuT = 0;
    double maxVariance = 0;

    // Build histogram (256 bins for grayscale)
    int hist[256] = {0};
    int totalPixels = src.rows * src.cols;

    for (int i = 0; i < src.rows; i++){
        uchar* spt = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++){
            hist[spt[j]]++;
        }
    }

    // Compute total mean
    double totalSum = 0;
    for (int t = 0; t < 256; t++){
        totalSum += t * hist[t];
    }

    // Sweep through all thresholds, track the one with max between-class variance
    double sumB = 0;  
    double weightB = 0;     

    for (int t = 0; t < 256; t++){
        weightB += hist[t];
        if (weightB == 0) continue;

        int weightF = totalPixels - weightB;
        if (weightF == 0) break;

        sumB += t * hist[t];

        double meanB = sumB / weightB;
        double meanF = (totalSum - sumB) / weightF;

        // Between-class variance
        double variance = (double)weightB * weightF * (meanB - meanF) * (meanB - meanF);

        if (variance > maxVariance){
            maxVariance = variance;
            otsuT = t;
        }
    }

    // Apply threshold (object = white, background = black)
    for (int i = 0; i < src.rows; i++){
        uchar* spt = src.ptr<uchar>(i);
        uchar* dpt = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++){
            if (spt[j] > otsuT){
                dpt[j] = 0;
            } else {
                dpt[j] = 255;
            }
        }
    }

    return 0;
}

int clean_binary(cv::Mat &src, cv::Mat &dst){
    // cleans image

    src.copyTo(dst);

    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));

   
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 2);

    
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);

    return 0;
}

int findLargestRegion(const cv::Mat& stats){
    // function finds the max region id 

    int maxLabel = 0;
    int maxArea = 0;
    int numLabels = stats.rows;

    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea)
        {
            maxArea = area;
            maxLabel = i;
        }
    }
    return maxLabel;
}

cv::Mat displayRegionMap(const cv::Mat& labels, const cv::Mat& stats, int minArea){
    // displays regions as an image using a color pallete

    int numLabels = stats.rows;

    std::vector<int> newLabel(numLabels, 0);
    int currentLabel = 1;

    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= minArea)
        {
            newLabel[i] = currentLabel;
            currentLabel++;
        }
    }

    std::vector<cv::Vec3b> colorTable(currentLabel + 1);
    cv::RNG rng(12345);
    for (int i = 1; i < currentLabel; i++)
    {
        colorTable[i] = cv::Vec3b(
            rng.uniform(0,255),
            rng.uniform(0,255),
            rng.uniform(0,255)
        );
    }

    cv::Mat regionMap(labels.size(), CV_8UC3, cv::Scalar(0,0,0));

    for (int y = 0; y < labels.rows; y++)
    {
        const int* labelPtr = labels.ptr<int>(y);
        cv::Vec3b* outPtr = regionMap.ptr<cv::Vec3b>(y);

        for (int x = 0; x < labels.cols; x++)
        {
            int originalLabel = labelPtr[x];
            int mappedLabel = newLabel[originalLabel];

            if (mappedLabel > 0)
                outPtr[x] = colorTable[mappedLabel];
            else
                outPtr[x] = cv::Vec3b(0,0,0);
        }
    }

    return regionMap;
}

RegionFeatures computeRegionFeatures(const cv::Mat& labels, int regionID){
    // computes features for a region ID

    RegionFeatures f;

    cv::Mat mask = (labels == regionID);
    cv::Moments m = cv::moments(mask, true);

    f.area = m.m00;

    f.centroid.x = m.m10 / m.m00;
    f.centroid.y = m.m01 / m.m00;

    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;

    f.orientation = 0.5 * atan2(2 * mu11, mu20 - mu02);

    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);

    f.obb = cv::minAreaRect(pts);

    double w = f.obb.size.width;
    double h = f.obb.size.height;

    f.aspectRatio = std::max(w, h) / std::min(w, h);
    f.percentFilled = f.area / (w * h);

    return f;
}

void drawPrimaryAxis(cv::Mat& img, const RegionFeatures& f){
    double length = 100;

    cv::Point2d p1(
        f.centroid.x + length * cos(f.orientation),
        f.centroid.y + length * sin(f.orientation)
    );
    cv::Point2d p2(
        f.centroid.x - length * cos(f.orientation),
        f.centroid.y - length * sin(f.orientation)
    );

    cv::line(img, p1, p2, cv::Scalar(0,255,0), 2);
}

void drawOBB(cv::Mat& img, const RegionFeatures& f){
    cv::Point2f vertices[4];
    f.obb.points(vertices);

    for (int i = 0; i < 4; i++)
    {
        cv::line(img, vertices[i],
                 vertices[(i+1)%4],
                 cv::Scalar(0,0,255), 2);
    }
}

void saveFeatureToFile(const std::string& filename,const std::string& label,const FeatureVector& v){
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Error opening database file\n";
        return;
    }

    file << label << ","
         << v.percentFilled << ","
         << v.aspectRatio << "\n";

    file.close();
    std::cout << "Saved feature vector for label: " << label << std::endl;
}

std::vector<DBEntry> loadDatabase(const std::string& filename){
    std::vector<DBEntry> database;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open DB file (will create on first save)\n";
        return database;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        DBEntry entry;

        std::getline(ss, entry.label, ',');

        std::getline(ss, token, ',');
        entry.features.percentFilled = std::stod(token);

        std::getline(ss, token, ',');
        entry.features.aspectRatio = std::stod(token);

        database.push_back(entry);
    }

    return database;
}

FeatureVector computeStdDev(const std::vector<DBEntry>& db){
    FeatureVector stddev = {1.0, 1.0};  

    int n = db.size();
    if (n < 2) return stddev;  

    FeatureVector mean = {0, 0};

    for (const auto& e : db)
    {
        mean.percentFilled += e.features.percentFilled;
        mean.aspectRatio   += e.features.aspectRatio;
    }

    mean.percentFilled /= n;
    mean.aspectRatio   /= n;

    FeatureVector variance = {0, 0};
    for (const auto& e : db)
    {
        variance.percentFilled += pow(e.features.percentFilled - mean.percentFilled, 2);
        variance.aspectRatio   += pow(e.features.aspectRatio   - mean.aspectRatio, 2);
    }

    stddev.percentFilled = sqrt(variance.percentFilled / n);
    stddev.aspectRatio   = sqrt(variance.aspectRatio   / n);

    
    if (stddev.percentFilled < 1e-9) stddev.percentFilled = 1.0;
    if (stddev.aspectRatio   < 1e-9) stddev.aspectRatio   = 1.0;

    return stddev;
}

FeatureVector extractFeatureVector(const RegionFeatures& f){
    FeatureVector v;
    v.percentFilled = f.percentFilled;
    v.aspectRatio = f.aspectRatio;
    return v;
}

double computeDistance(const FeatureVector& a,const FeatureVector& b,const FeatureVector& stddev){
    double d0 = (a.percentFilled - b.percentFilled) / stddev.percentFilled;
    double d1 = (a.aspectRatio   - b.aspectRatio)   / stddev.aspectRatio;
    return sqrt(d0*d0 + d1*d1);
}

std::string classifyObject(const FeatureVector& unknown,const std::vector<DBEntry>& db,const FeatureVector& stddev){
    double minDist = 1e9;
    double threshold_dist = 2.5;
    std::string bestLabel = "Unknown";

    for (const auto& entry : db)
    {
        double dist = computeDistance(unknown, entry.features, stddev);
        if (dist < minDist)
        {
            minDist = dist;

            if(minDist < threshold_dist){
                bestLabel = entry.label;
            }
        }
    }

    return bestLabel;
}


AxisExtents computeAxisExtents(const cv::Mat& labels, int regionID,const RegionFeatures& f){
    AxisExtents ext = {0, 0, 0, 0};

    double cosT = cos(f.orientation);
    double sinT = sin(f.orientation);

    cv::Mat mask = (labels == regionID);

    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);

    for (const auto& p : pts)
    {
        // Translate relative to centroid
        double dx = p.x - f.centroid.x;
        double dy = p.y - f.centroid.y;

        // Project onto primary axis (cos, sin) and secondary axis (-sin, cos)
        float proj1 = (float)(dx * cosT + dy * sinT);
        float proj2 = (float)(-dx * sinT + dy * cosT);

        if (proj1 < ext.minE1) ext.minE1 = proj1;
        if (proj1 > ext.maxE1) ext.maxE1 = proj1;
        if (proj2 < ext.minE2) ext.minE2 = proj2;
        if (proj2 > ext.maxE2) ext.maxE2 = proj2;
    }

    return ext;
}

void saveEmbeddingToFile(const std::string& filename,const std::string& label,const cv::Mat& embedding){

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Error opening embedding database file\n";
        return;
    }

    file << label;
    for (int i = 0; i < embedding.cols; i++)
    {
        file << "," << embedding.at<float>(0, i);
    }
    file << "\n";
    file.close();

    std::cout << "Saved embedding for label: " << label << std::endl;
}

std::vector<EmbeddingEntry> loadEmbeddingDatabase(const std::string& filename){
    std::vector<EmbeddingEntry> db;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open embedding DB (will create on first save)\n";
        return db;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        EmbeddingEntry entry;
        std::getline(ss, entry.label, ',');

        std::vector<float> values;
        while (std::getline(ss, token, ','))
        {
            values.push_back(std::stof(token));
        }

        entry.embedding = cv::Mat(1, (int)values.size(), CV_32F);
        for (int i = 0; i < (int)values.size(); i++)
        {
            entry.embedding.at<float>(0, i) = values[i];
        }

        db.push_back(entry);
    }

    return db;
}

double computeEmbeddingDistance(const cv::Mat& a, const cv::Mat& b)
{
    // Cosine distance: 1 - (a · b) / (|a| * |b|)
    double dot = a.dot(b);
    double normA = cv::norm(a);
    double normB = cv::norm(b);

    if (normA < 1e-9 || normB < 1e-9) return 1.0;

    return 1.0 - (dot / (normA * normB));
}

std::string classifyByEmbedding(const cv::Mat& unknown,const std::vector<EmbeddingEntry>& db){
    double minDist = 1e9;
    double threshold_dist = 0.4;
    std::string bestLabel = "Unknown";

    for (const auto& entry : db)
    {
        double dist = computeEmbeddingDistance(unknown, entry.embedding);
        if (dist < minDist)
        {
            minDist = dist;

            if(minDist < threshold_dist){
                bestLabel = entry.label;
            }
        }   
    }

    return bestLabel;
}