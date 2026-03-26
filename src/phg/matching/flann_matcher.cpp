#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
   index_params = flannKdTreeIndexParams(4);
   search_params = flannKsTreeSearchParams(40);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    int descs = query_desc.rows;
    matches.clear();
    matches.resize(descs);

    cv::Mat indices(descs, k, CV_32S);
    cv::Mat distances2(descs, k, CV_32F);

    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    for (int i = 0; i < descs; i++) {
        matches[i].reserve(k);
        for (int j = 0; j < k; j++) {
            matches[i].emplace_back(i, indices.at<int>(i, j), std::sqrt(distances2.at<float>(i, j)));
        }
    }
}
