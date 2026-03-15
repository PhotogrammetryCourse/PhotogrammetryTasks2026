#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include "flann_matcher.h"
#include "flann_factory.h"

static constexpr int N_TREES = 4;
static constexpr int N_CHECKS = 40;

phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
   index_params = flannKdTreeIndexParams(N_TREES);
   search_params = flannKsTreeSearchParams(N_CHECKS);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat idx(query_desc.rows, k, CV_32SC1);
    cv::Mat dst(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, idx, dst, k, *search_params);
    matches.resize(query_desc.rows);
    for (int i = 0; i < matches.size(); i++) {
        matches[i].reserve(k);
        for (int j = 0; j < k; ++j) {
            matches[i].push_back(cv::DMatch(i, idx.at<int>(i, j), dst.at<float>(i, j)));
        }
    }
}
