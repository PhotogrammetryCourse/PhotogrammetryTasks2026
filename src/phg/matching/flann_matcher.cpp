#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
    train_size = train_desc.rows;
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    if (!flann_index) {
        throw std::runtime_error("FlannMatcher:: knnMatch : matcher is not trained");
    }

    std::cout << "FlannMatcher::knnMatch : n query desc : " << query_desc.rows << ", n train desc : " << train_size << std::endl;

    const int ndesc = query_desc.rows;
    cv::Mat indices_query(ndesc, k, CV_32SC1);
    cv::Mat distances2_query(ndesc, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices_query, distances2_query, k, *search_params);
     
    matches.resize(ndesc);

    for (int qi = 0; qi < ndesc; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.clear();

        for (int col = 0; col < k; ++col) {
            cv::DMatch match;
            match.distance = std::sqrt(distances2_query.at<float>(qi, col));
            match.imgIdx = 0;
            match.queryIdx = qi;
            match.trainIdx = indices_query.at<int>(qi, col);
            dst.push_back(match);
        }
    }
}


