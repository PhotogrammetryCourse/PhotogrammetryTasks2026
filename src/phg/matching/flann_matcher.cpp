#include <algorithm>
#include <cmath>
#include "flann_matcher.h"
#include "flann_factory.h"

phg::FlannMatcher::FlannMatcher()
    : index_params(flannKdTreeIndexParams(kNtreesKD))
    , search_params(flannKsTreeSearchParams(kNChecks))
{}


void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const {
    matches.clear();
    if (query_desc.rows == 0) {
        return;
    }

    cv::Mat nearestIdx, squaredDists;
    flann_index->knnSearch(query_desc, nearestIdx, squaredDists, k, *search_params);

    matches.resize(query_desc.rows);

    auto buildMatchesForRow = [&](int rowIdx) {
        std::vector<cv::DMatch> rowMatches;
        rowMatches.reserve(k);
        const int* idxRow = nearestIdx.ptr<int>(rowIdx);
        const float* distRow = squaredDists.ptr<float>(rowIdx);
        for (int col = 0; col < k; ++col) {
            rowMatches.emplace_back(rowIdx, idxRow[col], 0, std::sqrt(std::max(0.0f, distRow[col])));
        }
        return rowMatches;
    };

    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i] = buildMatchesForRow(i);
    }
}
