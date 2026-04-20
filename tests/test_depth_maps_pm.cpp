#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <libutils/rasserts.h>
#include <libutils/string_utils.h>
#include <libutils/timer.h>

#include <phg/mvs/depth_maps/pm_depth_maps.h>
#include <phg/mvs/depth_maps/pm_geometry.h>
#include <phg/utils/cameras_bundler_import.h>
#include <phg/utils/point_cloud_export.h>

#include "utils/test_utils.h"

//________________________________________________________________________________
// Datasets:

// достаточно чтобы у вас работало на этом датасете, тестирование на Travis CI тоже ведется на нем
#define DATASET_DIR "saharov32"
#define DATASET_DOWNSCALE 4

// #define DATASET_DIR                  "temple47"
// #define DATASET_DOWNSCALE            2

// скачайте картинки этого датасета в папку data/src/datasets/herzjesu25/ по ссылке из файла LINK.txt в папке датасета
// #define DATASET_DIR                  "herzjesu25"
// #define DATASET_DOWNSCALE            8
//________________________________________________________________________________

TEST(test_depth_maps_pm, SingleDepthMap)
{
    Dataset dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);

    size_t cameras_limit = 5; // учитывая первые пять фотографий датасета, т.е. две камеры слева и две камеры справа
    dataset = dataset.subset(0, cameras_limit);
    phg::PMDepthMapsBuilder builder(dataset.ncameras, dataset.cameras_imgs, dataset.cameras_imgs_grey, dataset.cameras_labels, dataset.cameras_P, dataset.calibration);

    size_t ci = 2; // строим карту глубины для третьей камеры (индексация с нуля)
    cv::Mat depth_map, normal_map, cost_map;
    builder.buildDepthMap(ci, depth_map, normal_map, cost_map, dataset.cameras_depth_min[ci], dataset.cameras_depth_max[ci]);

    EXPECT_EQ(depth_map.type(), CV_32FC1);
    EXPECT_EQ(normal_map.type(), CV_32FC3);
    EXPECT_EQ(cost_map.type(), CV_32FC1);
    EXPECT_EQ(depth_map.rows, dataset.calibration.height());
    EXPECT_EQ(depth_map.cols, dataset.calibration.width());
}

TEST(test_depth_maps_pm, AllDepthMaps)
{
    // TODO этот код можно раскомментировать чтобы построить много карт глубины и сохранить их облака точек:
    // Dataset full_dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);
    //
    // const size_t ref_camera_shift = 2;
    // const size_t to_shift = 5;
    //
    // std::vector<cv::Vec3d> all_points;
    // std::vector<cv::Vec3b> all_colors;
    // std::vector<cv::Vec3d> all_normals;
    //
    // size_t ndepth_maps = 0;
    //
    // for (size_t from = 0; from + to_shift <= full_dataset.ncameras; ++from) {
    //     size_t to = from + to_shift;
    //
    //     Dataset dataset = full_dataset.subset(from, to);
    //
    //     phg::PMDepthMapsBuilder builder(dataset.ncameras, dataset.cameras_imgs, dataset.cameras_imgs_grey, dataset.cameras_labels, dataset.cameras_P, dataset.calibration);
    //     cv::Mat depth_map, normal_map, cost_map;
    //     builder.buildDepthMap(ref_camera_shift, depth_map, normal_map, cost_map, dataset.cameras_depth_min[ref_camera_shift], dataset.cameras_depth_max[ref_camera_shift]);
    //     phg::PMDepthMapsBuilder::buildGoodPoints(depth_map, normal_map, cost_map, dataset.cameras_imgs[ref_camera_shift], dataset.calibration, builder.getCameraPtoWorld(ref_camera_shift), all_points, all_colors, all_normals);
    //     ++ndepth_maps;
    //
    //     std::string tie_points_filename = std::string("data/debug/test_depth_maps_pm/") + getTestName() + "/all_points_" + to_string(ndepth_maps) + ".ply";
    //     phg::exportPointCloud(all_points, tie_points_filename, all_colors, all_normals);
    // }
}
