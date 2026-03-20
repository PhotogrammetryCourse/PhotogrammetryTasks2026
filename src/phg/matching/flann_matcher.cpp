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
    matches.clear();
    
    if (query_desc.empty() || !flann_index) {
        return; 
    }
    
    cv::Mat indices(query_desc.rows, k, CV_32S);
    cv::Mat distances(query_desc.rows, k, CV_32F);
    
    flann_index->knnSearch(query_desc, indices, distances, k, *search_params);
    
    matches.resize(query_desc.rows);
    
    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i].reserve(k);
        
        const int* indices_ptr = indices.ptr<int>(i);
        const float* distances_ptr = distances.ptr<float>(i);
        
        for (int j = 0; j < k; ++j) {
            int trainIdx = indices_ptr[j];
            if (trainIdx >= 0) {
                matches[i].emplace_back(i, trainIdx, std::sqrt(distances_ptr[j]));
            }
        }
    }
}
