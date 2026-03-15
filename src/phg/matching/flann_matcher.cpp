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
    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat dists(query_desc.rows, k, CV_32FC1);

    // do k-nearest neighbor search
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    // store results
    matches.resize(query_desc.rows);
    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i].reserve(k);
        for (int j = 0; j < k; ++j) {
            matches[i].emplace_back(i, indices.at<int>(i, j),  std::sqrt(dists.at<float>(i, j))  );
        }
    }

    // matches.resize(query_desc.rows);
    // for (int qi = 0; qi < query_desc.rows; ++qi) {
    //     std::vector<cv::DMatch> &dst = matches[qi];
    //     dst.resize(k);
    //     for (int ki = 0; ki < k; ++ki) {
    //         cv::DMatch match;
    //         match.imgIdx = 0;
    //         match.queryIdx = qi;
    //         match.trainIdx = indices.at<int>(qi, ki);
    //         match.distance = std::sqrt(dists.at<float>(qi, ki));
    //         dst[ki] = match;
    //     }
    // }


}
