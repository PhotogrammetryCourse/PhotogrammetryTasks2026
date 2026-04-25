#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    const int n_query = query_desc.rows;
    cv::Mat indices(n_query, k, CV_32SC1);
    cv::Mat dists(n_query, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    matches.resize(n_query);
    for (int i = 0; i < n_query; ++i) {
        matches[i].clear();
        for (int j = 0; j < k; ++j) {
            int idx = indices.at<int>(i, j);
            float dist = dists.at<float>(i, j);
            if (idx >= 0) {
                matches[i].emplace_back(i, idx, dist);
            }
        }
    }
}
