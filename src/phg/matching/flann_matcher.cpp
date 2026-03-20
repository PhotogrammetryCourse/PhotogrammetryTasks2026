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
    cv::Mat idx, dst;
    flann_index->knnSearch(query_desc, idx, dst, k, *search_params);
    matches.resize(query_desc.rows);
    for (int i = 0; i != idx.rows; ++i) {
        auto& row = matches[i];
        row.reserve(k);
        for (int j = 0; j != idx.cols; ++j)
            row.emplace_back(i, idx.at<int>(i, j), std::sqrt(dst.at<float>(i, j)));
    }
}
