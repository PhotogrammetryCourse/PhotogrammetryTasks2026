#include "descriptor_matcher.h"

#include "flann_factory.h"
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <set>

 /* @TUNABLE */
static constexpr float RATIO_THRESHOLD = 0.7;
static constexpr int SEARCH_CHECKS = 10;

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>>& matches, std::vector<cv::DMatch>& filtered_matches)
{
    filtered_matches.clear();
    const cv::DMatch* heap[3];
    auto cmp = [](const cv::DMatch* a, const cv::DMatch* b) { return a->distance < b->distance; };
    for (auto& match_row : matches) {
        int size = 0;
        for (auto& match : match_row) {
            heap[size++] = &match;
            std::push_heap(heap, heap + size, cmp);
            if (size == 3)
                std::pop_heap(heap, heap + size--, cmp);
        }
        if (size == 2) {
            std::pop_heap(heap, heap + size, cmp);
            if (heap[0]->distance / heap[1]->distance < RATIO_THRESHOLD * RATIO_THRESHOLD) {
                filtered_matches.push_back(*heap[0]);
            }
        } else if (size == 1) {
            filtered_matches.push_back(*heap[0]);
        }
    }
}

void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint> keypoints_query, const std::vector<cv::KeyPoint> keypoints_train, std::vector<cv::DMatch>& filtered_matches)
{
    filtered_matches.clear();

    const size_t total_neighbours = 5; // total number of neighbours to test (including candidate)
    const size_t consistent_matches = 3; // minimum number of consistent matches (including candidate)
    const float radius_limit_scale = 2.f; // limit search radius by scaled median

    const int n_matches = matches.size();

    if (n_matches < total_neighbours) {
        throw std::runtime_error("DescriptorMatcher::filterMatchesClusters : too few matches");
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        points_query.at<cv::Point2f>(i) = keypoints_query[matches[i].queryIdx].pt;
        points_train.at<cv::Point2f>(i) = keypoints_train[matches[i].trainIdx].pt;
    }

    // размерность всего 2, так что точное KD-дерево
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(1);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(SEARCH_CHECKS);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

    // для каждой точки найти total neighbors ближайших соседей
    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);

    // оценить радиус поиска для каждой картинки
    // NB: radius2_query, radius2_train: квадраты радиуса!
    float radius2_query, radius2_train;
    {
        std::vector<double> max_dists2_query(n_matches);
        std::vector<double> max_dists2_train(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            max_dists2_query[i] = distances2_query.at<float>(i, total_neighbours - 1);
            max_dists2_train[i] = distances2_train.at<float>(i, total_neighbours - 1);
        }

        int median_pos = n_matches / 2;
        std::nth_element(max_dists2_query.begin(), max_dists2_query.begin() + median_pos, max_dists2_query.end());
        std::nth_element(max_dists2_train.begin(), max_dists2_train.begin() + median_pos, max_dists2_train.end());

        radius2_query = max_dists2_query[median_pos] * radius_limit_scale * radius_limit_scale;
        radius2_train = max_dists2_train[median_pos] * radius_limit_scale * radius_limit_scale;
    }

    //    метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) имеют как минимум consistent_matches общих элементов
    // заполнить filtered_matches

    for (int i = 0; i < n_matches; ++i) {
        int n_consistent = 0;
        for (int src = 0; src < total_neighbours; ++src) {
            float dsrc = distances2_query.at<float>(i, src);
            if (dsrc > radius2_query)
                continue;

            for (int dst = 0; dst < total_neighbours; ++dst) {
                float ddst = distances2_train.at<float>(i, dst);
                if (ddst > radius2_train)
                    continue;

                if (indices_query.at<int>(i, src) == indices_train.at<int>(i, dst)) {
                    n_consistent += 1;

                    if (n_consistent >= consistent_matches)
                        goto early_success;
                }
            }
        }
        if (n_consistent >= consistent_matches) {
        early_success:
            filtered_matches.push_back(matches[i]);
        }
    }
}
