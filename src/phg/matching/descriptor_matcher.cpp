#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    /*
    The ratio test in descriptor matching is a technique used in computer vision to filter out false or ambiguous matches between features in two images. Proposed by David Lowe alongside the SIFT algorithm, it ensures that a match is valid by checking if the best match is significantly better than the second-best match. 

        Here is a breakdown of how it works:
        1. Mechanism
        K-Nearest Neighbors (KNN): For a descriptor in Image A, the matching algorithm finds the two closest descriptors in Image B using Euclidean distance (
        norm) or Hamming distance. Let these be 
        (best match) and 
        (second-best match).
        The Ratio: The ratio is calculated as: r = d2 / d1

        Thresholding: A threshold (usually 0.7 or 0.8) is set. A match is accepted only if: r < threshold
    */
    filtered_matches.clear();

    const float ratio_threshold = 0.6f; // scale50 test forced me to lower the threshold to 0.6, otherwise there were too few matches after filtering

    for (const auto &match_pair : matches) {
        if (match_pair.size() < 2) {
            // Skip if there are fewer than 2 matches for the query
            continue;
        }

        const cv::DMatch &best_match = match_pair[0];
        const cv::DMatch &second_best_match = match_pair[1];

        // Apply the ratio test !!!!
        if (best_match.distance < ratio_threshold * second_best_match.distance) {
            filtered_matches.push_back(best_match);
        }
    }
}


void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{

    /*
    here we got already foltered matched (after ratio test) and we want to filter them by clusters. 
    The idea is that correct matches should form spatially consistent clusters in both images,
     while incorrect matches are more likely to be randomly distributed.
    
    */
    filtered_matches.clear();

    const size_t  total_neighbours  = 15;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 5;  // minimum number of consistent matches (including candidate)
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
   std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(1);
   std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(std::max(64, n_matches));

    // when we call kd tree index , we "train" it on the points, 
    // so it builds the tree structure for fast nearest neighbor search.
    // my intuition :: its like database to search in 
   std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
   std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

   // для каждой точки найти total neighbors ближайших соседей
   cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
   cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
   cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
   cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    // now we know for each point in query and train, who are their total_neighbours nearest neighbors and what are the distances to them.
   index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
   index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);

   // оценить радиус поиска для каждой картинки
   // NB: radius2_query, radius2_train: квадраты радиуса!
   float radius2_query, radius2_train;
   {
       std::vector<double> max_dists2_query(n_matches);
       std::vector<double> max_dists2_train(n_matches);
       for (int i = 0; i < n_matches; ++i) {
        // here we collect the distances to the farthest neighbor (the total_neighbours-th neighbor) for each point, 
        // and then we will use the median of these distances to set a search radius.
           max_dists2_query[i] = distances2_query.at<float>(i, total_neighbours - 1);
           max_dists2_train[i] = distances2_train.at<float>(i, total_neighbours - 1);
       }

       int median_pos = n_matches / 2;
       std::nth_element(max_dists2_query.begin(), max_dists2_query.begin() + median_pos, max_dists2_query.end());
       std::nth_element(max_dists2_train.begin(), max_dists2_train.begin() + median_pos, max_dists2_train.end());

        /* example
        radius2_query = 120 means that for each point in the query dataset, neighbors within a squared distance of 120 will be considered for spatial consistency checks.
        */
       radius2_query = max_dists2_query[median_pos] * radius_limit_scale * radius_limit_scale;
       radius2_train = max_dists2_train[median_pos] * radius_limit_scale * radius_limit_scale;
   }

    //    метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) 
    // имеют как минимум consistent_matches общих элементов

    for (int i = 0; i < n_matches; ++i) {
         int count_consistent = 0; // we gonna keep track of how many neighbors are consistent between the query and train sets for the current match.
         for (int j_query = 0; j_query < total_neighbours; ++j_query) {
              if (distances2_query.at<float>(i, j_query) > radius2_query) {
                break; // remove neighbors that are too far in the query set
              }
    
              int idx_query = indices_query.at<int>(i, j_query);
              for (int j_train = 0; j_train < total_neighbours; ++j_train) {
                if (distances2_train.at<float>(i, j_train) > radius2_train) {
                     break; // exclude neighbors that are too far in the train set
                }
    
                int idx_train = indices_train.at<int>(i, j_train);
                if (idx_query == idx_train) {
                     ++count_consistent;
                     break;
                }
              }
         }
    
         if (count_consistent >= consistent_matches) {
              filtered_matches.push_back(matches[i]);
         }
    }

}
