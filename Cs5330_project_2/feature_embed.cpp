/*
Ghailan Fadah
1/28/26
CS5330
Program creates different feature embeddings for images
*/

#include "opencv2/opencv.hpp"
#include <cmath>
#include "csv_util.h"
#include "distance_metrics.h"
struct MatchResult {
    std::string filename;
    float dis;
};


int baseline_matching(cv::Mat &src, cv::Mat &res) {

    res.create(7, 7, CV_32FC3);

    int start_row = src.rows / 2 - 3;
    int start_col = src.cols / 2 - 3;

    for(int i = 0; i < 7; i++) {
        cv::Vec3b *spt = src.ptr<cv::Vec3b>(start_row + i);
        cv::Vec3f *rpt = res.ptr<cv::Vec3f>(i);

        for(int j = 0; j < 7; j++) {
            for(int c = 0; c < 3; c++) {
                rpt[j][c] = spt[start_col + j][c];
            }
        }
    }

    return 0;
}

int hist_matching(cv::Mat &src, cv::Mat &res) {

    res.create(16, 16, CV_32FC1);
    res.setTo(cv::Scalar(0));

    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *spt = src.ptr<cv::Vec3b>(i);

        for(int j = 0; j < src.cols; j++) {
            
            float B = spt[j][0];
            float G = spt[j][1];
            float R = spt[j][2];

            float div = R+G+B;
            div = div > 0.0 ? div : 1.0;

            float r = R / div;
            float g = G / div;

            int inx_r = static_cast<int>(r * 16);
            int inx_g = static_cast<int>(g * 16);

            inx_r = std::min(inx_r, 15);
            inx_g = std::min(inx_g, 15);

            res.at<float>(inx_r,inx_g)++;
            
        }
    }
    int total_pixels = src.rows * src.cols;
    res /= static_cast<float>(total_pixels);

    return 0;
}

int hist_top_matching(cv::Mat &src, cv::Mat &res) {

    res.create(8, 8, CV_32FC1);
    res.setTo(cv::Scalar(0));

    for(int i = 0; i < src.rows/2; i++) {
        cv::Vec3b *spt = src.ptr<cv::Vec3b>(i);

        for(int j = 0; j < src.cols; j++) {
            
            float B = spt[j][0];
            float G = spt[j][1];
            float R = spt[j][2];

            float div = R+G+B;
            div = div > 0.0 ? div : 1.0;

            float r = R / div;
            float g = G / div;

            int inx_r = static_cast<int>(r * 8);
            int inx_g = static_cast<int>(g * 8);

            inx_r = std::min(inx_r, 7);
            inx_g = std::min(inx_g, 7);

            res.at<float>(inx_r,inx_g)++;
            
        }
    }
    int total_pixels = (src.rows / 2)* src.cols;
    res /= static_cast<float>(total_pixels);

    return 0;
}

int hist_bottom_matching(cv::Mat &src, cv::Mat &res) {

    res.create(8, 8, CV_32FC1);
    res.setTo(cv::Scalar(0));

    for(int i = src.rows/2; i < src.rows; i++) {
        cv::Vec3b *spt = src.ptr<cv::Vec3b>(i);

        for(int j = 0; j < src.cols; j++) {
            
            float B = spt[j][0];
            float G = spt[j][1];
            float R = spt[j][2];

            float div = R+G+B;
            div = div > 0.0 ? div : 1.0;

            float r = R / div;
            float g = G / div;

            int inx_r = static_cast<int>(r * 8);
            int inx_g = static_cast<int>(g * 8);

            inx_r = std::min(inx_r, 7);
            inx_g = std::min(inx_g, 7);

            res.at<float>(inx_r,inx_g)++;
            
        }
    }
    int total_pixels = (src.rows / 2)* src.cols;
    res /= static_cast<float>(total_pixels);

    return 0;
}

int hist_top_bottom_matching(cv::Mat &src, cv::Mat &res) {


    cv::Mat res_bottom;
    cv::Mat res_top;
    hist_bottom_matching(src, res_bottom);
    hist_top_matching(src, res_top);
    
    cv::hconcat(res_top.reshape(1,1),res_bottom.reshape(1,1),res);

    res /= cv::sum(res)[0];
  
    return(0);

}

int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    /*
    function create a custom sobel 3X3 filter in x-direction using pointers to access pxs
    [-1,0,1
    -2,0,2
    -1,0,1]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i<src.rows-1; i++){
        cv::Vec3b *s_pt_up = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *s_pt_down = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *d_pt = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j<src.cols-1; j++){

            for(int c=0;c<3;c++){

                int res = s_pt_up[j-1][c]*-1 + s_pt_up[j][c]*0 + s_pt_up[j+1][c]*1
                + s_pt_mid[j-1][c]*-2 + s_pt_mid[j][c]*0 + s_pt_mid[j+1][c]*2
                + s_pt_down[j-1][c]*-1 + s_pt_down[j][c]*0 + s_pt_down[j+1][c]*1;

                // range [-255-255]
                d_pt[j][c] = res;

            }
        }
    }
    return(0);
}

int sobelY3x3( cv::Mat &src, cv::Mat &dst){
    /*
    function create a custom sobel 3X3 filter in y-direction using pointers to access pxs
    [1,2,1
    0,0,0
    -1,-2,-1]
    input: cv::Mat &src, cv::Mat &dst
    output <int> 0
    */

    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i<src.rows-1; i++){
        cv::Vec3b *s_pt_up = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s_pt_mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *s_pt_down = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *d_pt = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j<src.cols-1; j++){

            for(int c=0;c<3;c++){

                int res = s_pt_up[j-1][c]*1 + s_pt_up[j][c]*2 + s_pt_up[j+1][c]*1
                + s_pt_mid[j-1][c]*0 + s_pt_mid[j][c]*0 + s_pt_mid[j+1][c]*0
                + s_pt_down[j-1][c]*-1 + s_pt_down[j][c]*-2 + s_pt_down[j+1][c]*-1;

                // range [-255-255]
                d_pt[j][c] = res;

            }
        }
    }

    return (0);

}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
    /*
    function determines mag of sx, sy and claps mag val to [0-255] range
    
    input: cv::Mat &sx, cv::Mat &sy, cv::Mat &dst
    output <int> 0
    */


    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i<sx.rows; i++){
        cv::Vec3s *px = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *py = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *pd = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j<sx.cols; j++){

             for(int c=0;c<3;c++){

                int mag = std::sqrt(px[j][c]*px[j][c] + py[j][c]*py[j][c]);
                pd[j][c] = cv::saturate_cast<uchar>(mag);
             }

        }
    }

    return(0);

}

int hist_texture(cv::Mat &src, cv::Mat &res) {

    cv::Mat sx; cv::Mat sy; cv::Mat mag;
    sobelY3x3(src, sy);
    sobelX3x3(src,sx);
    magnitude(sx,sy,mag);

    res.create(16, 16, CV_32FC1);
    res.setTo(cv::Scalar(0));

    for(int i = 0; i < mag.rows; i++) {
        cv::Vec3b *spt = mag.ptr<cv::Vec3b>(i);

        for(int j = 0; j < mag.cols; j++) {
            
            int gray = (spt[j][0] + spt[j][1] + spt[j][2]) / 3;
            int bin = gray * 16 / 256;
            bin = std::min(bin, 15);

            res.at<float>(bin,0)++;
        }
    }
    
    float s = 0.0f;
    for (int i = 0; i < res.total(); i++) {
        s += res.at<float>(i);
        }   
    res /= s;

    return 0;
}

int hist_texture_rg_matching(cv::Mat &src, cv::Mat &res){

    cv::Mat res_text;
    cv::Mat res_rg;
    hist_matching(src, res_rg);
    hist_texture(src, res_text);

    res.create(512, 1, CV_32FC1);
    res.setTo(cv::Scalar(0));
    
    cv::hconcat(res_rg.reshape(1,1),res_text.reshape(1,1),res);

    float s = 0.0f;
    for (int i = 0; i < res.total(); i++) {
        s += res.at<float>(i);
        }
    res /= s;
  
    return(0);

}

int run_csv_matching(const char* target_image,
                     const char* csv_file,
                     int N)
{
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    if (read_image_data_csv((char*)csv_file, filenames, data, 0) != 0) {
        std::cerr << "Failed to read CSV file\n";
        return -1;
    }

    // Find target index
    int target_idx = -1;
    for (size_t i = 0; i < filenames.size(); i++) {
        if (strcmp(filenames[i], target_image) == 0) {
            target_idx = i;
            break;
        }
    }

    if (target_idx < 0) {
        std::cerr << "Target image not found in CSV\n";
        return -1;
    }

    const std::vector<float>& target_feat = data[target_idx];

    auto cmp = [](const MatchResult& a, const MatchResult& b) {
        return a.dis < b.dis;   // max heap
    };

    std::priority_queue<
        MatchResult,
        std::vector<MatchResult>,
        decltype(cmp)
    > heap(cmp);

    for (size_t i = 0; i < data.size(); i++) {
        float d = 0.0f;

        /*
        for (size_t k = 0; k < target_feat.size(); k++) {
            float diff = target_feat[k] - data[i][k];
            d += diff * diff;
        }
        */
        
        d = cosine_dis(target_feat, data[i]);

        if (heap.size() < N) {
            heap.push({filenames[i], d});
        }
        else if (d < heap.top().dis) {
            heap.pop();
            heap.push({filenames[i], d});
        }
    }

        std::vector<MatchResult> topMatches;

    while (!heap.empty()) {
        topMatches.push_back(heap.top());
        heap.pop();
    }

    std::reverse(topMatches.begin(), topMatches.end());

    std::cout << "Top matches for: " << target_image << "\n";
    for (size_t i = 0; i < topMatches.size(); i++) {
        std::cout << i + 1 << ". "
                  << topMatches[i].filename
                  << " | Dist = "
                  << topMatches[i].dis
                  << "\n";
    }

    return 0;
}

int sunset_matching(cv::Mat &src, cv::Mat &res) {

    res.create(16, 16, CV_32FC1);
    res.setTo(cv::Scalar(0));

    // Focus on sky region (top half)
    for(int i = 0; i < src.rows / 2; i++) {
        cv::Vec3b *spt = src.ptr<cv::Vec3b>(i);

        for(int j = 0; j < src.cols; j++) {

            float B = spt[j][0];
            float G = spt[j][1];
            float R = spt[j][2];

           
            float warmth = R - B; 
            float warmth_norm = (warmth + 255.0f) / 510.0f;

            float maxc = std::max({R, G, B});
            float minc = std::min({R, G, B});
            float sat = (maxc > 0.0f) ? (maxc - minc) / maxc : 0.0f;

            int w_bin = static_cast<int>(warmth_norm * 16);
            int s_bin = static_cast<int>(sat * 16);

            w_bin = std::min(std::max(w_bin, 0), 15);
            s_bin = std::min(std::max(s_bin, 0), 15);

            res.at<float>(w_bin, s_bin)++;
        }
    }

    res /= static_cast<float>((src.rows / 2) * src.cols);
    return 0;
}






