#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    train_desc_ = train_desc.clone();
    flann_index = flannKdTreeIndex(train_desc_, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    const int n_query_desc = query_desc.rows;
    cv::Mat indices(n_query_desc, k, CV_32SC1);
    cv::Mat distances2(n_query_desc, k, CV_32FC1);

    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    matches.assign(n_query_desc, {});
    for (int qi = 0; qi < n_query_desc; ++qi) {
        matches[qi].reserve(k);
        for (int ki = 0; ki < k; ++ki) {
            const int train_idx = indices.at<int>(qi, ki);
            const float distance = std::sqrt(distances2.at<float>(qi, ki));
            matches[qi].emplace_back(qi, train_idx, distance);
        }
    }
}
