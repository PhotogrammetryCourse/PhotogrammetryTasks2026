#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    //    index_params = flannKdTreeIndexParams(DONE);
    //    search_params = flannKsTreeSearchParams(DONE);
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{

    cv::Mat index, dist2;
    flann_index->knnSearch(query_desc, index, dist2, k, *search_params);
    matches.resize(query_desc.rows);
    for (int i = 0; i < query_desc.rows; i++) {
        matches[i].clear();
        for (int j = 0; j < k; j++) {
            int id_train = index.at<int>(i, j);
            float dist = dist2.at<float>(i, j);
            matches[i].emplace_back(i, id_train, dist);
        }
    }
//    throw std::runtime_error("not implemented yet");
}
