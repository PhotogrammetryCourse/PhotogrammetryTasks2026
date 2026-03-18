#include <iostream>
#include <algorithm>
#include <cmath>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
//    index_params = flannKdTreeIndexParams(TODO);
//    search_params = flannKsTreeSearchParams(TODO);
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    if (!flann_index) {
        throw std::runtime_error("FlannMatcher:: knnMatch : matcher is not trained");
    }
    if (k <= 0) {
        throw std::runtime_error("FlannMatcher:: knnMatch : k must be > 0");
    }
    if (query_desc.empty()) {
        matches.clear();
        return;
    }

    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat distances2(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    matches.resize(query_desc.rows);
    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i].clear();
        matches[i].reserve(k);
        for (int j = 0; j < k; ++j) {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = indices.at<int>(i, j);
            m.imgIdx = 0;
            m.distance = std::sqrt(std::max(0.f, distances2.at<float>(i, j)));
            matches[i].push_back(m);
        }
    }
}
