#pragma once

#include <opencv2/core.hpp>

namespace phg {

    cv::Mat estimateHomography4Points(const cv::Point2f &l0, const cv::Point2f &l1,
                                      const cv::Point2f &l2, const cv::Point2f &l3,
                                      const cv::Point2f &r0, const cv::Point2f &r1,
                                      const cv::Point2f &r2, const cv::Point2f &r3);

    cv::Mat findHomography(const std::vector<cv::Point2f> &points_lhs,
                               const std::vector<cv::Point2f> &points_rhs);

    cv::Mat findHomographyCV(const std::vector<cv::Point2f> &points_lhs,
                           const std::vector<cv::Point2f> &points_rhs);

    cv::Point2d transformPoint(const cv::Point2d &pt, const cv::Mat &T);
    cv::Point2d transformPointCV(const cv::Point2d &pt, const cv::Mat &T);
}
