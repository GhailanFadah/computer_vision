/*
Ghailan Fadah
1/28/26
CS5330
Program takes in an image file, a directory, matching type, feature type, and number of top mathcing images to return.
return top N images to the target image
*/

#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include <functional>
#include "feature_embed.h"
#include "distance_metrics.h"

int main(int argc, char *argv[]) {

  // check for sufficient arguments
  if( argc < 6) {
    printf("usage: %s <Target image> <directory path> <feature type> <mathcing type> <number of images>\n", argv[0]);
    exit(-1);
  }

  char dirname[256];
  char target[256];
  char buffer[256];
  std::string feature_type = argv[3];
  std::string matching_type = argv[4];
  strcpy(target, argv[1]);
  strcpy(dirname, argv[2]);
  int N = std::atoi(argv[5]);
 
  DIR *dirp;
  struct dirent *dp;
 
  cv::Mat tar;
  cv::Mat img;
  cv::Mat feat_s;
  cv::Mat feat_r;


  // comparator for queue
  auto cmp = [](const MatchResult& a, const MatchResult& b) {
    return a.dis < b.dis;
    };

  // priority queue that hold MatchResult
  std::priority_queue<MatchResult, std::vector<MatchResult>, decltype(cmp)> heap(cmp);
  
  // function variables
  std::function<int(cv::Mat&, cv::Mat&)> func_feature;
  std::function<float(cv::Mat&, cv::Mat&)> func_distance;

  if(feature_type == "hist"){
    func_feature = hist_matching;
  }else if (feature_type == "2hist"){
    func_feature = hist_top_bottom_matching;
  }else if (feature_type == "hist-text"){
    func_feature = hist_texture_rg_matching;
  }else if (feature_type == "sun"){
    func_feature = sunset_matching;
  }else if (feature_type == "dnn"){
    return run_csv_matching(target, dirname, N);
  }else{
    func_feature = baseline_matching;
  }

  if(matching_type == "i"){
    func_distance = Inter_Dis;
  }else{
    func_distance = SSD;
  }
  

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }


  tar = cv::imread(target);
  func_feature(tar, feat_s);
  
  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if(strstr(dp->d_name, ".jpg") ||
	     strstr(dp->d_name, ".png") ||
	     strstr(dp->d_name, ".ppm") ||
	     strstr(dp->d_name, ".tif") ) {

      

        // build the overall filename
        strcpy(buffer, dirname);
        strcat(buffer, "/");
        strcat(buffer, dp->d_name);


        img = cv::imread(buffer);
        func_feature(img, feat_r);

        float distance = func_distance(feat_s, feat_r);

        if(heap.size() < N) {
          heap.push({buffer, distance});
          }
        else if (distance < heap.top().dis) {
          heap.pop();                  // remove worst match
          heap.push({buffer, distance});      // insert better one
          }

    }
  }

  std::vector<MatchResult> topMatches;

  while (!heap.empty()) {
    topMatches.push_back(heap.top());
    heap.pop();
  }

  std::reverse(topMatches.begin(), topMatches.end());
  
  //std::cout << "Top matches for:"<<target<< std::endl;

  for (size_t i = 0; i < topMatches.size(); i++) {
    std::cout 
              << topMatches[i].filename << ","
              << topMatches[i].dis
              << std::endl;
  }

  return(0);
}


