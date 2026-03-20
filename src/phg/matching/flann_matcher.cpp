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
    cv::Mat indices, dists;
    flann_index->knnSearch(query_desc, indices, dists, k);

    matches.resize(query_desc.rows);
    for (size_t i = 0; i != indices.rows; ++i) {
        auto &row = matches[i];
        row.reserve(k);

        for (size_t j = 0; j != indices.cols; j++) {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = indices.at<int>(i, j);
            match.distance = std::sqrt(dists.at<float>(i, j));
            row.push_back(match);
        }
    }
}
