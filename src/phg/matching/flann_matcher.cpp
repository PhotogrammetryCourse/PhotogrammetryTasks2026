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
    int n = query_desc.rows;
    cv::Mat inds(n, k, CV_32SC1);
    cv::Mat dists(n, k, CV_32FC1);
    flann_index -> knnSearch(query_desc, inds, dists, k, *search_params);
    matches.resize(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (int ind = inds.at<int>(i, j); ind > 0) {
                matches[i].emplace_back(i, ind, 0, std::sqrt(dists.at<float>(i, j)));
            }
        }
    }
}
