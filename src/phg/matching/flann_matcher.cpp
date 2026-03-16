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
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    if (!flann_index)
        throw std::runtime_error("not trained");
    if (query_desc.empty()) {
        matches.clear();
        return;
    }

    cv::Mat indexes(query_desc.rows, k, CV_32SC1);
    cv::Mat distances2(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indexes, distances2, k, *search_params);

    matches.assign(query_desc.rows, {});
    for (int qi = 0; qi < query_desc.rows; ++qi) {
        matches[qi].reserve(k);
        for (int ki = 0; ki < k; ++ki) {
            int train_idx = indexes.at<int>(qi, ki);
            float dist = std::sqrt(distances2.at<float>(qi, ki));
            matches[qi].emplace_back(qi, train_idx, 0, dist);
        }
    }
}
