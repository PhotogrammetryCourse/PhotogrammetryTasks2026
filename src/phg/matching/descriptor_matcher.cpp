#include <iterator>
#include <unordered_set>
#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"
#include <libutils/rasserts.h>

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();
    const double k_max_ratio = 0.7;

    for (const auto& match : matches) {
        if (match[0].distance < k_max_ratio * match[1].distance) {
            filtered_matches.push_back(match[0]);
        }
    }
}


void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    const size_t  total_neighbours  = 5;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 3;  // minimum number of consistent matches (including candidate)
    const float  radius_limit_scale  = 2.f;  // limit search radius by scaled median

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
    static constexpr std::size_t kNtreesKD = 4;
    static constexpr std::size_t kNChecks = 32;
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(kNtreesKD);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(kNChecks);

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
    std::vector<cv::DMatch> consistent;
    for (int i = 0; i < n_matches; ++i) {
        int shared_count = 0;
        for (size_t q = 0; q < total_neighbours; ++q) {
            if (distances2_query.at<float>(i, q) > radius2_query) {
                continue;
            }
            int idx_q = indices_query.at<int>(i, q);
            for (size_t t = 0; t < total_neighbours; ++t) {
                if (distances2_train.at<float>(i, t) > radius2_train) {
                    continue;
                }
                if (idx_q == indices_train.at<int>(i, t)) {
                    shared_count++;
                    break;
                }
            }
        }
        if (shared_count >= (int)consistent_matches) {
            consistent.push_back(matches[i]);
        }
    }

    std::sort(consistent.begin(), consistent.end(), [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

    // Guarantee uniqueness of coordinates and indexes
    std::vector<bool> query_used(keypoints_query.size(), false);
    std::vector<bool> train_used(keypoints_train.size(), false);

    for (const auto& m : consistent) {
        if (!query_used[m.queryIdx] && !train_used[m.trainIdx]) {
            bool coord_duplicate = false;
            for (const auto& f : filtered_matches) {
                if (cv::norm(keypoints_query[m.queryIdx].pt - keypoints_query[f.queryIdx].pt) < 1e-4 || cv::norm(keypoints_train[m.trainIdx].pt - keypoints_train[f.trainIdx].pt) < 1e-4) {
                    coord_duplicate = true;
                    break;
                }
            }

            if (!coord_duplicate) {
                filtered_matches.push_back(m);
                query_used[m.queryIdx] = true;
                train_used[m.trainIdx] = true;
            }
        }
    }
}
