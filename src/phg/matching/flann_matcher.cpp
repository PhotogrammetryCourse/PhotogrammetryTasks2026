#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat dists(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, dists,k, *search_params);
    matches.resize(query_desc.rows);
    for(int qd_idx = 0; qd_idx< query_desc.rows; qd_idx++){
       std::vector<cv::DMatch> &d_matches = matches[qd_idx];
        d_matches.resize(k);
       for (int i =0; i< k; i++){
            cv::DMatch match(qd_idx, indices.at<int>(qd_idx, i),0,std::sqrt(dists.at<float>(qd_idx, i)));
            d_matches[i] = match;
       }
    }
}
