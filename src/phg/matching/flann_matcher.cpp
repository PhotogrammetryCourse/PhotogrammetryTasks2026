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
    if (!flann_index) {
        throw std::runtime_error("FlannMatcher::knnMatch : matcher is not trained");
    }

    if (k != 2) {
        throw std::runtime_error("FlannMatcher::knnMatch : only k = 2 supported");
    }

    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat dists2(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, dists2, k, *search_params);

    matches.resize(query_desc.rows);
    for (int qi = 0; qi < query_desc.rows; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.resize(k);
        for (int ki = 0; ki < k; ++ki) {
            cv::DMatch match;
            match.imgIdx = 0;
            match.queryIdx = qi;
            match.trainIdx = indices.at<int>(qi, ki);
            match.distance = std::sqrt(dists2.at<float>(qi, ki));
            dst[ki] = match;
        }
    }
}
