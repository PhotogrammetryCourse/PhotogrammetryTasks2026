#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <fstream>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>
#include <phg/sfm/ematrix.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/sfm/defines.h>
#include <phg/sfm/triangulation.h>
#include <phg/sfm/resection.h>
#include <phg/utils/point_cloud_export.h>

#include <ceres/rotation.h>
#include <ceres/ceres.h>

#define ENABLE_BA                             1
#define NIMGS_LIMIT                           100
#define INTRINSICS_CALIBRATION_MIN_IMGS       5

#define ENABLE_INSTRINSICS_K1_K2              1
#define INTRINSIC_K1_K2_MIN_IMGS              7

#define ENABLE_OUTLIERS_FILTRATION_3_SIGMA    1
#define ENABLE_OUTLIERS_FILTRATION_COLINEAR   1
#define ENABLE_OUTLIERS_FILTRATION_NEGATIVE_Z 1

//________________________________________________________________________________
// Datasets:

#define DATASET_DIR                  "saharov32"
#define DATASET_DOWNSCALE            1
#define DATASET_F                    (1585.5 / DATASET_DOWNSCALE)

//#define DATASET_DIR                  "herzjesu25"
//#define DATASET_DOWNSCALE            2
//#define DATASET_F                    (2761.5 / DATASET_DOWNSCALE)

// но temple47 - не вышло, я не разобрался в чем с ним проблема, может быть слишком мало точек, может критерии фильтрации выкидышей для него слишком строги
//#define DATASET_DIR                  "temple47"
//#define DATASET_DOWNSCALE            1
//#define DATASET_F                    (1520.4 / DATASET_DOWNSCALE) // see temple47/README.txt about K-matrix (i.e. focal length = K11 from templeR_par.txt)

// Специальный датасет прямо с Марса!
/*
#define DATASET_DIR                  "perseverance25"
#define DATASET_DOWNSCALE            1
#define DATASET_F                    (4720.4 / DATASET_DOWNSCALE)
// на этом датасете фотографии длиннофокусные, поэтому многие лучи почти колинеарны, поэтому этот фильтр подавляет все точки и третья камера не подвыравнивается
#undef  ENABLE_OUTLIERS_FILTRATION_COLINEAR
#define ENABLE_OUTLIERS_FILTRATION_COLINEAR 0
 */
// и в целом все плохо... у меня не получилось выравнять этот датасет нашим простым прототипом
//________________________________________________________________________________


namespace {

    vector3d relativeOrientationAngles(const matrix3d &R0, const vector3d &O0, const matrix3d &R1, const vector3d &O1) {
        vector3d a = R0 * vector3d{0, 0, 1};
        vector3d b = O0 - O1;
        vector3d c = R1 * vector3d{0, 0, 1};

        double norma = cv::norm(a);
        double normb = cv::norm(b);
        double normc = cv::norm(c);

        if (norma == 0 || normb == 0 || normc == 0) {
            throw std::runtime_error("norma == 0 || normb == 0 || normc == 0");
        }

        a /= norma;
        b /= normb;
        c /= normc;

        vector3d cos_vals;

        cos_vals[0] = a.dot(c);
        cos_vals[1] = a.dot(b);
        cos_vals[2] = b.dot(c);

        return cos_vals;
    }

    // one track corresponds to one 3d point
    class Track {
    public:
        Track()
        {
            disabled = false;
        }

        bool disabled;
        std::vector<std::pair<int, int>> img_kpt_pairs;
    };

}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors);

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose=false);

TEST (SFM, ReconstructNViews) {
    using namespace cv;

    // Чтобы было проще - картинки упорядочены заранее в файле data/src/datasets/DATASETNAME/ordered_filenames.txt
    std::vector<cv::Mat> imgs;
    std::vector<std::string> imgs_labels;
    {
        std::ifstream in(std::string("data/src/datasets/") + DATASET_DIR + "/ordered_filenames.txt");
        size_t nimages = 0;
        in >> nimages;
        std::cout << nimages << " images" << std::endl;
        for (int i = 0; i < nimages; ++i) {
            std::string img_name;
            in >> img_name;
            std::string img_path = std::string("data/src/datasets/") + DATASET_DIR + "/" + img_name;
            cv::Mat img = cv::imread(img_path);

            if (img.empty()) {
                throw std::runtime_error("Can't read image: " + to_string(img_path));
            }

            // выполняем уменьшение картинки если оригинальные картинки в этом датасете - слишком большие для используемой реализации SIFT
            int downscale = DATASET_DOWNSCALE;
            while (downscale > 1) {
                cv::pyrDown(img, img);
                rassert(downscale % 2 == 0, 1249219412940115);
                downscale /= 2;
            }

            imgs.push_back(img);
            imgs_labels.push_back(img_name);
        }
    }

    phg::Calibration calib(imgs[0].cols, imgs[0].rows);
    calib.f_ = DATASET_F;

    // сверяем что все картинки одинакового размера (мы ведь предполагаем что их снимала одна и та же камера с одними и те же интринсиками)
    for (const auto &img : imgs) {
        rassert(img.cols == imgs[0].cols && img.rows == imgs[0].rows, 34125412512512);
    }

    const size_t n_imgs = std::min(imgs.size(), (size_t) NIMGS_LIMIT);

    std::cout << "detecting points..." << std::endl;
    std::vector<std::vector<cv::KeyPoint>> keypoints(n_imgs);
    std::vector<std::vector<int>> track_ids(n_imgs);
    std::vector<cv::Mat> descriptors(n_imgs);
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    for (int i = 0; i < (int) n_imgs; ++i) {
        detector->detectAndCompute(imgs[i], cv::noArray(), keypoints[i], descriptors[i]);
        track_ids[i].resize(keypoints[i].size(), -1);
    }

    std::cout << "matching points..." << std::endl;
    using Matches = std::vector<cv::DMatch>;
    std::vector<std::vector<Matches>> matches(n_imgs);
    size_t ndone = 0;
    #pragma omp parallel for
    for (int i = 0; i < n_imgs; ++i) {
        matches[i].resize(n_imgs);
        for (int j = 0; j < n_imgs; ++j) {
            if (i == j) {
                continue;
            }

            // Flann matching
            std::vector<std::vector<DMatch>> knn_matches;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->knnMatch( descriptors[i], descriptors[j], knn_matches, 2 );
            std::vector<DMatch> good_matches(knn_matches.size());
            for (int k = 0; k < (int) knn_matches.size(); ++k) {
                good_matches[k] = knn_matches[k][0];
            }

            // Filtering matches GMS
            std::vector<DMatch> good_matches_gms;
            int inliers = phg::filterMatchesGMS(good_matches, keypoints[i], keypoints[j], imgs[i].size(), imgs[j].size(), good_matches_gms, false);
            #pragma omp critical
            {
                ++ndone;
                if (inliers > 0) {
                    std::cout << to_percent(ndone, n_imgs * (n_imgs - 1)) + "% - Cameras " << i << "-" << j << " (" << imgs_labels[i] << "-" << imgs_labels[j] << "): " << inliers << " matches" << std::endl;
                }
            }

            matches[i][j] = good_matches_gms;
        }
    }

    std::vector<Track> tracks;
    std::vector<vector3d> tie_points;
    std::vector<matrix34d> cameras(n_imgs);
    std::vector<char> aligned(n_imgs);

    // align first two cameras
    {
        std::cout << "Initial alignment from cameras #0 and #1 (" << imgs_labels[0] << ", " << imgs_labels[1] << ")" << std::endl;
        // matches from first to second image in specified sequence
        const Matches &good_matches_gms = matches[0][1];
        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[0];
        const std::vector<cv::KeyPoint> &keypoints1 = keypoints[1];
        const phg::Calibration &calib0 = calib;
        const phg::Calibration &calib1 = calib;

        std::vector<cv::Vec2d> points0, points1;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints0[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints1[match.trainIdx].pt;
            points0.push_back(pt1);
            points1.push_back(pt2);
        }

        matrix3d F = phg::findFMatrix(points0, points1, 3, false);
        matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

        matrix34d P0, P1;
        phg::decomposeEMatrix(P0, P1, E, points0, points1, calib0, calib1, false);

        cameras[0] = P0;
        cameras[1] = P1;
        aligned[0] = true;
        aligned[1] = true;

        matrix34d Ps[2] = {P0, P1};
        for (int i = 0; i < (int) good_matches_gms.size(); ++i) {
            vector3d ms[2] = {calib0.unproject(points0[i]), calib1.unproject(points1[i])};
            vector4d X = phg::triangulatePoint(Ps, ms, 2);

            if (X(3) == 0) {
                std::cerr << "infinite point" << std::endl;
                continue;
            }

            vector3d X3d{X(0) / X(3), X(1) / X(3), X(2) / X(3)};

            tie_points.push_back(X3d);

            Track track;
            track.img_kpt_pairs.push_back({0, good_matches_gms[i].queryIdx});
            track.img_kpt_pairs.push_back({1, good_matches_gms[i].trainIdx});
            track_ids[0][good_matches_gms[i].queryIdx] = tracks.size();
            track_ids[1][good_matches_gms[i].trainIdx] = tracks.size();
            tracks.push_back(track);
        }

        int ncameras = 2;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

#if ENABLE_BA
        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
#endif
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }

    // append remaining cameras one by one
    for (int i_camera = 2; i_camera < n_imgs; ++i_camera) {

        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[i_camera];
        const phg::Calibration &calib0 = calib;

        std::vector<vector3d> Xs;
        std::vector<vector2d> xs;
        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id != -1) {
                    if (tracks[track_id].disabled)
                        continue; // пропускаем выключенные точки (признанные выбросами)
                    Xs.push_back(tie_points[track_id]);
                    cv::Vec2f pt = keypoints0[match.queryIdx].pt;
                    xs.push_back(pt);
                }
            }
        }

        std::cout << "Append camera #" << i_camera << " (" << imgs_labels[i_camera] << ") to alignment via " << Xs.size() << " common points" << std::endl;
        rassert(Xs.size() > 0, 2318254129859128305);
        matrix34d P = phg::findCameraMatrix(calib0, Xs, xs, false);

        cameras[i_camera] = P;
        aligned[i_camera] = true;

        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const std::vector<cv::KeyPoint> &keypoints1 = keypoints[i_camera_prev];
            const phg::Calibration &calib1 = calib;
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id == -1) {
                    matrix34d Ps[2] = {P, cameras[i_camera_prev]};
                    cv::Vec2f pts[2] = {keypoints0[match.queryIdx].pt, keypoints1[match.trainIdx].pt};
                    vector3d ms[2] = {calib0.unproject(pts[0]), calib1.unproject(pts[1])};
                    vector4d X = phg::triangulatePoint(Ps, ms, 2);

                    if (X(3) == 0) {
                        std::cerr << "infinite point" << std::endl;
                        continue;
                    }

                    tie_points.push_back({X(0) / X(3), X(1) / X(3), X(2) / X(3)});

                    Track track;
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track.img_kpt_pairs.push_back({i_camera_prev, match.trainIdx});
                    track_ids[i_camera][match.queryIdx] = tracks.size();
                    track_ids[i_camera_prev][match.trainIdx] = tracks.size();
                    tracks.push_back(track);
                } else {
                    if (tracks[track_id].disabled)
                        continue; // пропускаем выключенные точки (признанные выбросами)
                    Track &track = tracks[track_id];
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track_ids[i_camera][match.queryIdx] = track_id;
                }
            }
        }

        int ncameras = i_camera + 1;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

        // Запуск Bundle Adjustment
#if ENABLE_BA
        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
#endif

        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }
}

class ReprojectionError {
public:
    ReprojectionError(double x, double y) : observed_x(x), observed_y(y)
    {}

    template <typename T>
    bool operator()(const T* camera_extrinsics,
                    const T* camera_intrinsics,
                    const T* point_global,
                    T* residuals) const {
        const T* translation = camera_extrinsics + 0;
        const T* rotation = camera_extrinsics + 3;

        T point_shifted[3] = {
                point_global[0] - translation[0],
                point_global[1] - translation[1],
                point_global[2] - translation[2]
        };
        T point_camera[3];
        ceres::AngleAxisRotatePoint(rotation, point_shifted, point_camera);

        T x = point_camera[0] / point_camera[2];
        T y = point_camera[1] / point_camera[2];

#if ENABLE_INSTRINSICS_K1_K2
        T r2 = x * x + y * y;
        T distortion = T(1.0) + camera_intrinsics[0] * r2 + camera_intrinsics[1] * r2 * r2;
        x *= distortion;
        y *= distortion;
#endif

        x *= camera_intrinsics[2];
        y *= camera_intrinsics[2];

        x += camera_intrinsics[3];
        y += camera_intrinsics[4];

        residuals[0] = x - T(observed_x);
        residuals[1] = y - T(observed_y);
        return true;
    }

protected:
    double observed_x;
    double observed_y;
};

void printCamera(double* camera_intrinsics)
{
    std::cout << "camera: k1=" << camera_intrinsics[0] << ", k2=" << camera_intrinsics[1] << ", "
              << "f=" << camera_intrinsics[2] << ", "
              << "cx=" << camera_intrinsics[3] << ", cy=" << camera_intrinsics[4] << std::endl;
}

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose)
{
    ceres::Problem problem;

    ASSERT_NEAR(calib.f_ , DATASET_F, 0.2 * DATASET_F);
    ASSERT_NEAR(calib.cx_, 0.0, 0.3 * calib.width());
    ASSERT_NEAR(calib.cy_, 0.0, 0.3 * calib.height());

    double camera_intrinsics[5] = {
            calib.k1_,
            calib.k2_,
            calib.f_,
            calib.cx_ + calib.width() * 0.5,
            calib.cy_ + calib.height() * 0.5
    };
    std::cout << "Before BA ";
    printCamera(camera_intrinsics);

    const int CAMERA_EXTRINSICS_NPARAMS = 6;

    std::vector<double> cameras_extrinsics(CAMERA_EXTRINSICS_NPARAMS * ncameras, 0.0);
    for (size_t camera_id = 0; camera_id < (size_t) ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* translation = camera_extrinsics + 0;
        double* rotation_angle_axis = camera_extrinsics + 3;

        matrix3d Rt = R.t();
        ceres::RotationMatrixToAngleAxis(&(Rt(0, 0)), rotation_angle_axis);

        for (int d = 0; d < 3; ++d) {
            translation[d] = O[d];
        }
    }

    const double sigma = 2.0;
    const double inlier_threshold2 = 9.0 * sigma * sigma;

    double inliers_mse = 0.0;
    size_t inliers = 0;
    size_t nprojections = 0;
    std::vector<double> cameras_inliers_mse(ncameras, 0.0);
    std::vector<size_t> cameras_inliers(ncameras, 0);
    std::vector<size_t> cameras_nprojections(ncameras, 0);

    std::vector<ceres::CostFunction*> reprojection_residuals;
    std::vector<ceres::CostFunction*> reprojection_residuals_for_deletion;

    for (size_t i = 0; i < tie_points.size(); ++i) {
        const Track &track = tracks[i];
        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;
            int keypoint_id = track.img_kpt_pairs[ci].second;
            cv::Vec2f px = keypoints[camera_id][keypoint_id].pt;

            ceres::CostFunction* keypoint_reprojection_residual = new ceres::AutoDiffCostFunction<ReprojectionError,
                    2,
                    6, 5, 3>
                    (new ReprojectionError(px[0], px[1]));
            reprojection_residuals.push_back(keypoint_reprojection_residual);

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
            double* point3d_params = &(tie_points[i][0]);

            {
                const double* params[3];
                double residual[2] = {-1.0, -1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < inlier_threshold2) {
                    inliers_mse += error2;
                    ++inliers;
                    cameras_inliers_mse[camera_id] += error2;
                    ++cameras_inliers[camera_id];
                }
                ++nprojections;
                ++cameras_nprojections[camera_id];
            }

            if (!track.disabled) {
                problem.AddResidualBlock(keypoint_reprojection_residual, new ceres::HuberLoss(3.0 * sigma),
                                         camera_extrinsics,
                                         camera_intrinsics,
                                         point3d_params);
            } else {
                reprojection_residuals_for_deletion.push_back(keypoint_reprojection_residual);
            }
        }
    }
    std::cout << "Before BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;
    for (size_t camera_id = 0; camera_id < (size_t) ncameras; ++camera_id) {
        size_t ninls = cameras_inliers[camera_id];
        size_t nproj = cameras_nprojections[camera_id];
        std::cout << "    Camera #" << camera_id << " projections: " << to_percent(ninls, nproj) << "% inliers "
                  << "(" << ninls << "/" << nproj << ") with MSE=" << (cameras_inliers_mse[camera_id] / ninls) << std::endl;
    }

    if (ncameras < INTRINSICS_CALIBRATION_MIN_IMGS) {
        problem.SetParameterBlockConstant(camera_intrinsics);
    } else if (ncameras < INTRINSIC_K1_K2_MIN_IMGS) {
        problem.SetManifold(camera_intrinsics, new ceres::SubsetManifold(5, {0, 1}));
    }

    {
        size_t camera_id = 0;
        double* camera0_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetParameterBlockConstant(camera0_extrinsics);
    }
    {
        size_t camera_id = 1;
        double* camera1_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetManifold(camera1_extrinsics, new ceres::SubsetManifold(6, {0, 1, 2}));
    }

    if (ENABLE_BA) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = verbose;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (verbose) {
            std::cout << summary.BriefReport() << std::endl;
        }
    }

    std::cout << "After BA ";
    printCamera(camera_intrinsics);
    calib.k1_ = camera_intrinsics[0];
    calib.k2_ = camera_intrinsics[1];
    calib.f_ = camera_intrinsics[2];
    calib.cx_ = camera_intrinsics[3] - calib.width() * 0.5;
    calib.cy_ = camera_intrinsics[4] - calib.height() * 0.5;

    ASSERT_NEAR(calib.f_ , DATASET_F, 0.2 * DATASET_F);
    ASSERT_NEAR(calib.cx_, 0.0, 0.3 * calib.width());
    ASSERT_NEAR(calib.cy_, 0.0, 0.3 * calib.height());

    for (size_t camera_id = 0; camera_id < (size_t) ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;

        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);
        std::cout << "Camera #" << camera_id << " center: " << O << " -> ";

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* translation = camera_extrinsics + 0;
        double* rotation_angle_axis = camera_extrinsics + 3;

        matrix3d Rt;
        ceres::AngleAxisToRotationMatrix(rotation_angle_axis, &(Rt(0, 0)));
        R = Rt.t();

        for (int d = 0; d < 3; ++d) {
            O[d] = translation[d];
        }

        std::cout << O << std::endl;
        cameras[camera_id] = phg::composeCameraMatrixRO(R, O);
    }

    inliers_mse = 0.0;
    inliers = 0;
    nprojections = 0;
    cameras_inliers_mse = std::vector<double>(ncameras, 0.0);
    cameras_inliers = std::vector<size_t>(ncameras, 0);
    cameras_nprojections = std::vector<size_t>(ncameras, 0);

    size_t n_old_outliers = 0;
    size_t n_new_outliers = 0;

    size_t next_loss_k = 0;
    const double max_parallel_cos = 0.9990482215818578;
    for (size_t i = 0; i < tie_points.size(); ++i) {
        Track &track = tracks[i];
        bool should_be_disabled = false;
        vector3d track_point = tie_points[i];

        if (ENABLE_OUTLIERS_FILTRATION_COLINEAR && ENABLE_BA && track.img_kpt_pairs.size() >= 2) {
            bool has_good_ray_pair = false;
            for (size_t p0 = 0; p0 < track.img_kpt_pairs.size() && !has_good_ray_pair; ++p0) {
                vector3d O0;
                matrix3d R0;
                phg::decomposeUndistortedPMatrix(R0, O0, cameras[track.img_kpt_pairs[p0].first]);
                vector3d ray0 = track_point - O0;
                double norm0 = cv::norm(ray0);
                if (norm0 == 0.0) {
                    continue;
                }
                ray0 /= norm0;

                for (size_t p1 = p0 + 1; p1 < track.img_kpt_pairs.size(); ++p1) {
                    vector3d O1;
                    matrix3d R1;
                    phg::decomposeUndistortedPMatrix(R1, O1, cameras[track.img_kpt_pairs[p1].first]);
                    vector3d ray1 = track_point - O1;
                    double norm1 = cv::norm(ray1);
                    if (norm1 == 0.0) {
                        continue;
                    }
                    ray1 /= norm1;

                    double cos_angle = ray0.dot(ray1);
                    if (cos_angle < max_parallel_cos) {
                        has_good_ray_pair = true;
                        break;
                    }
                }
            }
            if (!has_good_ray_pair) {
                should_be_disabled = true;
            }
        }

        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;

            ceres::CostFunction* keypoint_reprojection_residual = reprojection_residuals[next_loss_k++];

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
            double* point3d_params = &(tie_points[i][0]);

            matrix3d R;
            vector3d camera_origin;
            phg::decomposeUndistortedPMatrix(R, camera_origin, cameras[camera_id]);

            if (ENABLE_OUTLIERS_FILTRATION_NEGATIVE_Z && ENABLE_BA) {
                vector3d track_in_camera = R * (track_point - camera_origin);
                if (track_in_camera[2] < 0.0) {
                    should_be_disabled = true;
                }
            }

            {
                const double* params[3];
                double residual[2] = {-1.0, -1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < inlier_threshold2) {
                    inliers_mse += error2;
                    ++inliers;
                    cameras_inliers_mse[camera_id] += error2;
                    ++cameras_inliers[camera_id];
                } else if (ENABLE_OUTLIERS_FILTRATION_3_SIGMA && ENABLE_BA) {
                    should_be_disabled = true;
                }
                ++nprojections;
                ++cameras_nprojections[camera_id];
            }
        }

        if (should_be_disabled && !track.disabled) {
            track.disabled = true;
            ++n_new_outliers;
        } else if (track.disabled) {
            ++n_old_outliers;
        }
    }
    std::cout << "After BA tie poits: " << to_percent(n_old_outliers, tie_points.size()) << "% old + " << to_percent(n_new_outliers, tie_points.size()) << "% new = " << to_percent(n_old_outliers + n_new_outliers, tie_points.size()) << "% total outliers" << std::endl;
    std::cout << "After BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;
    for (size_t camera_id = 0; camera_id < (size_t) ncameras; ++camera_id) {
        size_t ninls = cameras_inliers[camera_id];
        size_t nproj = cameras_nprojections[camera_id];
        double mse = (cameras_inliers_mse[camera_id] / ninls);
        std::cout << "    Camera #" << camera_id << " projections: " << to_percent(ninls, nproj) << "% inliers "
                  << "(" << ninls << "/" << nproj << ") with MSE=" << mse << std::endl;
        ASSERT_GT(ninls, 0.15 * nproj);
    }

    for (auto ptr : reprojection_residuals_for_deletion) {
        delete ptr;
    }
}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 24152151251241);

    tie_points_and_cameras.clear();
    tie_points_colors.clear();

    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const Track &track = tracks[i];
        if (track.disabled)
            continue;

        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_and_cameras.push_back(tie_points[i]);
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    for (int i_camera = 0; i_camera < ncameras; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points_and_cameras.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points_and_cameras.push_back(O + R.t() * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }
}
