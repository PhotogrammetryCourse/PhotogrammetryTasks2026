#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // DONE параметры для приближенного поиска
   index_params = flannKdTreeIndexParams(4);
   search_params = flannKsTreeSearchParams(40);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices_query(query_desc.rows, k, CV_32SC1);
    cv::Mat distances2_query(query_desc.rows, k, CV_32FC1);

    flann_index->knnSearch(query_desc, indices_query, distances2_query, k, *search_params);

    matches.resize(query_desc.rows);
    for(size_t mIdx = 0; mIdx < query_desc.rows; ++mIdx) {
        matches[mIdx].clear();
        for(size_t i = 0; i < k; ++i) {
            matches[mIdx].emplace_back(mIdx, indices_query.at<int>(mIdx, i), distances2_query.at<float>(mIdx, i));
        }
    }
}
