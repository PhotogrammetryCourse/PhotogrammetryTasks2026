#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();
    for (const auto& nearest_matches : matches) {
        if (nearest_matches.size() > 1) {
            cv::DMatch best_match = std::min(nearest_matches[0], nearest_matches[1]);
            cv::DMatch second_best_match = std::max(nearest_matches[0], nearest_matches[1]);
            for (int i = 2; i < nearest_matches.size(); ++i) {
                if (nearest_matches[i] < best_match) {
                    second_best_match = std::exchange(best_match, nearest_matches[i]);
                } else if (nearest_matches[i] < second_best_match) {
                    second_best_match = nearest_matches[i];
                }
            }
            if (best_match.distance < 0.8f * second_best_match.distance) {
                filtered_matches.push_back(best_match);
            }
        } else if (!nearest_matches.empty()) {
            filtered_matches.push_back(nearest_matches[0]);
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
    // TODO заменить параметры
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(1);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(-1);

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

    // метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) 
    // имеют как минимум consistent_matches общих элементов
    // TODO заполнить filtered_matches
    
//    std::set<std::pair<int, int>> match_pairs;
//    for (int i = 0; i < n_matches; ++i) {
//        match_pairs.insert({matches[i].queryIdx, matches[i].trainIdx});
//    }
    
    for (int i = 0; i < n_matches; ++i) {
        int cnt_matches = 0;
        
        for (int query_col = 0; query_col < total_neighbours; ++query_col) {
            for (int train_col = 0; train_col < total_neighbours; ++train_col) {
                int query_index = indices_query.at<int>(i, query_col);
                int train_index = indices_train.at<int>(i, train_col);
                float query_dist2 = distances2_query.at<float>(i, query_col);
                float train_dist2 = distances2_train.at<float>(i, train_col);
                if (query_dist2 < radius2_query && train_dist2 < radius2_train && 
                        //match_pairs.find({query_index, train_index}) != match_pairs.end()) {
                        train_index == query_index) {
                    cnt_matches++;
                }
                    
            }
        }
        
        if (cnt_matches >= consistent_matches) {
            filtered_matches.push_back(matches[i]);
        }
    }
}
