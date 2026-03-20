#include <iostream>
#include <cmath>
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
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);

    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    matches.clear();
    matches.resize(query_desc.rows);

    if (query_desc.rows == 0) {
        return;
    }

    cv::Mat indices;
    cv::Mat distances;
    flann_index->knnSearch(query_desc, indices, distances, k);

    for (int qi = 0; qi < query_desc.rows; ++qi) {
        std::vector<cv::DMatch>& dst = matches[qi];
        dst.clear();
        dst.reserve(k);

        for (int j = 0; j < k; ++j) {
            int train_idx = indices.at<int>(qi, j);
            if (train_idx < 0) {
                continue;
            }

            cv::DMatch match;
            match.distance = std::sqrt(distances.at<float>(qi, j));
            match.imgIdx = 0;
            match.queryIdx = qi;
            match.trainIdx = train_idx;
            dst.push_back(match);
        }
    }
}
