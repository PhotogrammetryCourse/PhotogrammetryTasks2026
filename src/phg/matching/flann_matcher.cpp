#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(50);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    for (size_t i = 0; i < query_desc.size().height; i++) {
        std::vector<int> outIndices(k);
        std::vector<float> outDists(k);

        flann_index->knnSearch(query_desc.row(i), outIndices, outDists, k, *search_params);
        matches.push_back({cv::DMatch(i, outIndices[0], std::sqrt(outDists[0])), cv::DMatch(i, outIndices[1], std::sqrt(outDists[1]))});
    }
}
