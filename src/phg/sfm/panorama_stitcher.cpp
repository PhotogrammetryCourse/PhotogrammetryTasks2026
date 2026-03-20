#include "panorama_stitcher.h"
#include "homography.h"

#include <iostream>
#include <libutils/bbox2.h>

/*
 * imgs - список картинок
 * parent - список индексов, каждый индекс указывает, к какой картинке должна быть приклеена текущая картинка
 *          этот список образует дерево, корень дерева (картинка, которая ни к кому не приклеивается, приклеиваются только к ней), в данном массиве имеет значение -1
 * homography_builder - функтор, возвращающий гомографию по паре картинок
 * */
cv::Mat phg::stitchPanorama(const std::vector<cv::Mat>& imgs, const std::vector<int>& parent, std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>& homography_builder)
{
    const int n_images = imgs.size();

    // Compute homographies for each image relative to the root
    std::vector<cv::Mat> Hs(n_images);
    std::vector<bool> already_calculated(n_images, false);
        std::vector<cv::Mat> parent_Hs(n_images);
        auto calc_Hs = [&](auto&& self, const int& imgs_idx) -> const cv::Mat& {
            if (already_calculated[imgs_idx])
                return Hs[imgs_idx];
            if (parent[imgs_idx] < 0) {
                Hs[imgs_idx] = cv::Mat::eye(3, 3, CV_64FC1);
                already_calculated[imgs_idx] = true;
                return Hs[imgs_idx];
            }

            if (parent_Hs[imgs_idx].empty())
                parent_Hs[imgs_idx] = homography_builder(imgs[imgs_idx], imgs[parent[imgs_idx]]);
            
            Hs[imgs_idx] = self(self, parent[imgs_idx]) * parent_Hs[imgs_idx];
            already_calculated[imgs_idx] = true;
            return Hs[imgs_idx];
        };
        for (int i = 0; i < n_images; ++i) {
            calc_Hs(calc_Hs, i);
        }
    
    // Compute the bounding box for the panorama
    bbox2<double, cv::Point2d> bbox;
    for (int i = 0; i < n_images; ++i) {
        double w = imgs[i].cols;
        double h = imgs[i].rows;
        bbox.grow(phg::transformPoint(cv::Point2d(0.0, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, h), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(0, h), Hs[i]));
    }

    std::cout << "bbox: " << bbox.max() << ", " << bbox.min() << std::endl;

    int result_width = bbox.width() + 1;
    int result_height = bbox.height() + 1;

    cv::Mat result = cv::Mat::zeros(result_height, result_width, CV_8UC3);

    // из-за растяжения пикселей при использовании прямой матрицы гомографии после отображения между пикселями остается пустое пространство
    // лучше использовать обратную и для каждого пикселя на итоговвой картинке проверять, с какой картинки он может получить цвет
    // тогда в некоторых пикселях цвет будет дублироваться, но изображение будет непрерывным
    //    for (int i = 0; i < n_images; ++i) {
    //        for (int y = 0; y < imgs[i].rows; ++y) {
    //            for (int x = 0; x < imgs[i].cols; ++x) {
    //                cv::Vec3b color = imgs[i].at<cv::Vec3b>(y, x);

    //                cv::Point2d pt_dst = applyH(cv::Point2d(x, y), Hs[i]) - bbox.min();
    //                int y_dst = std::max(0, std::min((int) std::round(pt_dst.y), result_height - 1));
    //                int x_dst = std::max(0, std::min((int) std::round(pt_dst.x), result_width - 1));

    //                result.at<cv::Vec3b>(y_dst, x_dst) = color;
    //            }
    //        }
    //    }

    std::vector<cv::Mat> Hs_inv;
    std::transform(Hs.begin(), Hs.end(), std::back_inserter(Hs_inv), [&](const cv::Mat& H) { return H.inv(); });

    // Blend all images into the panorama
#pragma omp parallel for
    for (int y = 0; y < result_height; ++y) {
        for (int x = 0; x < result_width; ++x) {
            cv::Point2d pt_dst(x, y);

            // Test all images, pick the first valid one
            for (int i = 0; i < n_images; ++i) {
                cv::Point2d pt_src = phg::transformPoint(pt_dst + bbox.min(), Hs_inv[i]);

                int x_src = std::round(pt_src.x);
                int y_src = std::round(pt_src.y);

if (x_src >= 0 && x_src < imgs[i].cols && y_src >= 0 && y_src < imgs[i].rows) {
                    result.at<cv::Vec3b>(y, x) = imgs[i].at<cv::Vec3b>(y_src, x_src);
                    break;
                }
            }
        }
    }

    return result;
}
