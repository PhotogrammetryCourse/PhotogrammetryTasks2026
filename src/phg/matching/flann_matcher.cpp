#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"
#include "gms_matcher_impl.h"


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
    matches.clear();
    int n = query_desc.rows;
    cv::Mat indices_query(n, k, CV_32SC1);
    cv::Mat distances_query(n, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices_query, distances_query, k, *search_params);
    matches.resize(n);
    for (int i = 0; i < n; i++) {
        matches.reserve(k);
        for (int j = 0; j < k; j++) {
            int ind = indices_query.at<int>(i, j);
            if (ind < 0) continue;
            float dist = std::sqrt(distances_query.at<float>(i, j));
            matches[i].emplace_back(i, ind, 0, dist);

        }
        //std::sort(matches[i].begin(), matches[i].end(), [](const cv::DMatch a, const cv::DMatch b) {return a.distance < b.distance;});
    }
}
