#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libutils/rasserts.h>
#include <libutils/timer.h>

#include <filesystem>
#include <phg/sift/sift.h>

#include "utils/test_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SHOW_RESULTS 0 // если вам хочется сразу видеть результат в окошке - переключите в 1, но не забудьте выключить перед коммитом (иначе бот в CI будет ждать веками)
#define MAX_ACCEPTED_PIXEL_ERROR                                                                                                                                                                                                              \
    0.01 // максимальное расстояние в пикселях (процент от ширины картинки) между ключевыми точками чтобы их можно было зачесть как "почти совпавшие" (это очень завышенный порог, по-хорошему должно быть 0.005 например)
#define MAX_AVG_PIXEL_ERROR 0.075

#define GAUSSIAN_NOISE_STDDEV 1.0

#define ENABLE_MY_SIFT_TESTING 1

#define DENY_CREATE_REF_DATA 1

struct MatchingPairData {
    size_t npoints1, npoints2, nmatches;
};

template <typename SIFT> MatchingPairData evaluateMatching(SIFT& sift, const cv::Mat& img1, const cv::Mat& img2, const std::string& output_matches_img_path)
{
    std::vector<cv::KeyPoint> kpts1;
    cv::Mat desc1;
    sift.detectAndCompute(img1, { }, kpts1, desc1);

    std::vector<cv::KeyPoint> kpts2;
    cv::Mat desc2;
    sift.detectAndCompute(img2, { }, kpts2, desc2);

    // Brute-force matching with ratio test
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches) {
        if (m.size() == 2 && m[0].distance < ratioThresh * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }

    // RANSAC filtering with fundamental matrix
    std::vector<cv::DMatch> inlierMatches;
    if (goodMatches.size() >= 15) {
        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& m : goodMatches) {
            pts1.push_back(kpts1[m.queryIdx].pt);
            pts2.push_back(kpts2[m.trainIdx].pt);
        }

        std::vector<uchar> inlierMask;
        cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inlierMask);

        for (size_t i = 0; i < goodMatches.size(); i++) {
            if (inlierMask[i]) {
                inlierMatches.push_back(goodMatches[i]);
            }
        }
    }

    std::cout << "N keypoints: left " << kpts1.size() << ", right " << kpts2.size() << std::endl;
    std::cout << "Good matches:       " << goodMatches.size() << std::endl;
    std::cout << "Inlier matches:     " << inlierMatches.size() << std::endl;

    if (!output_matches_img_path.empty()) {
        cv::Mat imgMatches;
        cv::drawMatches(img1, kpts1, img2, kpts2, inlierMatches, imgMatches);
        cv::imwrite(output_matches_img_path, imgMatches);
    }

    return { kpts1.size(), kpts2.size(), inlierMatches.size() };
}

// функция рисует кружки случайного цвета вокруг точек, но если для точки не нашлось сопоставления - кружок будет толстый и ярко красный
void drawKeyPoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kps, const std::vector<unsigned char>& is_not_matched)
{
    cv::RNG r(124124);
    for (size_t i = 0; i < kps.size(); ++i) {
        int thickness = 1;
        cv::Scalar color;
        if (is_not_matched[i]) {
            color = CV_RGB(255, 0, 0); // OpenCV использует BGR схему вместо RGB, но можно использовать этот макрос вместо BGR - cv::Scalar(blue=0, green=0, red=255)
            thickness = 2;
        } else {
            color = cv::Scalar(r.uniform(0, 255), r.uniform(0, 255), 0);
        }
        int radius = std::max(2, (int)(kps[i].size / 5.0f));
        float angle = kps[i].angle;
        cv::circle(img, kps[i].pt, radius, color, thickness);
        if (angle != -1.0) {
            cv::line(img, kps[i].pt, cv::Point((int)std::round(kps[i].pt.x + radius * sin(angle * M_PI / 180.0)), (int)std::round(kps[i].pt.y + radius * cos(angle * M_PI / 180.0))), color);
        }
    }
}

// Функция ищет знаковый угол между двумя направлениями (по кратчайшему пути, т.е. результат от -180 до 180)
double diffAngles(double angle0, double angle1)
{
    if (angle0 != -1.0 && angle1 != -1.0) {
        rassert(angle0 >= 0.0 && angle0 < 360.0, 1235612352151);
        rassert(angle1 >= 0.0 && angle1 < 360.0, 4645315415);
        float diff;
        if ((angle1 <= angle0 + 180 && angle0 + 180 <= 360) || (angle1 >= angle0 - 180 && angle0 - 180 >= 0)) {
            diff = angle1 - angle0;
        } else if (angle1 > angle0 + 180 && angle0 + 180 <= 360) {
            diff = -(angle0 + (360 - angle1));
        } else if (angle1 <= angle0 - 180 && angle0 - 180 >= 0) {
            diff = (360 - angle0) + angle1;
        } else {
            rassert(false, 1234124125125135);
        }
        rassert(diff >= -180 && diff <= 180, 233536136131);
        return diff;
    } else {
        return 0.0;
    }
}

// На вход передается матрица описывающая преобразование картинки (сдвиг, поворот, масштабирование или их комбинация), допустимый процент Recall, и опционально можно тестировать другую картинку
void evaluateDetection(const cv::Mat& M, double minRecall, cv::Mat img0 = cv::Mat())
{
    if (img0.empty()) {
        img0 = cv::imread("data/src/test_sift/unicorn.png"); // грузим картинку по умолчанию
    }

    ASSERT_FALSE(img0.empty()); // проверка что картинка была загружена
    // убедитесь что рабочая папка (Edit Configurations...->Working directory) указывает на корневую папку проекта (и тогда картинка по умолчанию найдется по относительному пути - data/src/test_sift/unicorn.png)

    size_t width = img0.cols;
    size_t height = img0.rows;
    cv::Mat transformedImage;
    cv::warpAffine(img0, transformedImage, M, cv::Size(width, height)); // строим img1 - преобразованная исходная картинка в соответствии с закодированным в матрицу M искажением пространства
    cv::Mat noise(cv::Size(width, height), CV_8UC3);
    cv::setRNGSeed(125125); // фиксируем рандом для детерминизма (чтобы результат воспроизводился из раза в раз)
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(GAUSSIAN_NOISE_STDDEV));
    cv::add(transformedImage, noise, transformedImage); // добавляем к преобразованной картинке гауссиан шума
    cv::Mat img1 = transformedImage;

    {
        for (int method = 0; method < 3; ++method) { // тестируем три метода: OpenCV ORB, OpenCV SIFT, ваш SIFT
            std::vector<cv::KeyPoint> kps0;
            std::vector<cv::KeyPoint> kps1;

            cv::Mat desc0;
            cv::Mat desc1;

            timer t; // очень удобно встраивать профилирование вашего кода по мере его написания, тогда полную картину видеть гораздо проще (особенно это помогает со старым кодом)
            std::string method_name;
            std::string log_prefix;

            phg::SIFTParams p;
            p.nfeatures = 500;
            if (method == 0) {
                method_name = "ORB";
                log_prefix = "[ORB_OCV] ";
                // ORB - один из видов ключевых дескрипторов, отличается высокой скоростью и относительно неплохим качеством
                cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(p.nfeatures); // здесь можно было бы поиграть с его параметрами, например выделять больше чем 500 точек, строить большее число ступеней пирамиды и т.п.
                detector->detect(img0, kps0); // детектируем ключевые точки на исходной картинке
                detector->detect(img1, kps1); // детектируем ключевые точки на преобразованной картинке

                detector->compute(img0, kps0, desc0);
                detector->compute(img1, kps1, desc1);
            } else if (method == 1) {
                method_name = "SIFTOCV";
                log_prefix = "[SIFTOCV] ";
                cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create(
                    p.nfeatures, p.n_octave_layers, p.contrast_threshold, p.edge_threshold); // здесь можно было бы поиграть с его параметрами, например выделять больше чем 500 точек, строить большее число ступеней пирамиды и т.п.
                detector->detect(img0, kps0); // детектируем ключевые точки на исходной картинке
                detector->detect(img1, kps1); // детектируем ключевые точки на преобразованной картинке

                detector->compute(img0, kps0, desc0);
                detector->compute(img1, kps1, desc1);
            } else if (method == 2) {
#if ENABLE_MY_SIFT_TESTING
                method_name = "SIFT_MY";
                log_prefix = "[SIFT_MY] ";
                phg::SIFT mySIFT(p);
                mySIFT.detectAndCompute(img0, kps0, desc0);
                mySIFT.detectAndCompute(img1, kps1, desc1);
#else
                return;
#endif
            } else {
                rassert(false, 13532513412); // это не проверка как часть тестирования, это проверка что число итераций в цикле и if-else ветки все еще согласованы и не разошлись
            }

            std::cout << log_prefix << "Points detected: " << kps0.size() << " -> " << kps1.size() << " (in " << t.elapsed() << " sec)" << std::endl;

            std::vector<cv::Point2f> ps01(kps0.size()); // давайте построим эталон - найдем куда бы должны были сместиться ключевые точки с исходного изображения с учетом нашей матрицы трансформации M
            {
                std::vector<cv::Point2f> ps0(kps0.size()); // здесь мы сейчас расположим детектированные ключевые точки (каждую нужно преобразовать из типа КлючеваяТочка в Точка2Дэ)
                for (size_t i = 0; i < kps0.size(); ++i) {
                    ps0[i] = kps0[i].pt;
                }
                cv::transform(ps0, ps01, M); // преобразовываем все точки с исходного изображения в систему координат его искаженной версии с учетом матрицы M, эти точки - эталон
            }

            double error_sum = 0.0; // считаем суммарную ошибку координат сопоставлений точек чтобы найти среднюю ошибку (в пикселях)
            double size_ratio_sum = 0.0; // хотим найти среднее соотношение размера сопоставленных ключевых точек (чтобы сверить эту пропорцию с тестируемым перепадом масштаба)
            double angle_diff_sum = 0.0; // хотим найти среднее отличие угла наклона сопоставленных ключевых точек (чтобы сверить этот угол с тестируемым в тестах поворотом)
            double desc_dist_sum = 0.0; // хотим найти среднее расстояние между дескрипторами сопоставленных ключевых точек
            double desc_rand_dist_sum = 0.0; // найдем среднее расстояние между случайными парами ключевых точек (чтобы было с чем сравнить расстояние сопоставленных точек)
            size_t n_matched = 0; // число успешно сопоставившихся исходных точек
            size_t n_in_bounds = 0; // число исходных точек которые после преобразования координат не вышли за пределы картинки (т.е. в целом имели шансы на успешное сопоставление)
            std::vector<unsigned char> is_not_matched0(kps0.size(), true); // для каждой исходной точки хотим понять сопоставилась ли она
            std::vector<unsigned char> is_not_matched1(kps1.size(), true); // для каждой точки с результирующей картинки хотим понять сопоставился ли с ней хоть кто-то

            // эта прагма - способ распараллелить цикл на все ядра процессора (см. OpenMP parallel for)
            // reduction позволяет сказать OpenMP что нужно провести редукцию суммированием для каждой из переменных: error_sum, n_matched, n_in_bounds, ...
            // мы ведь хотим найти сумму по всем потокам
            #pragma omp parallel for reduction(+ : error_sum, n_matched, n_in_bounds, size_ratio_sum, angle_diff_sum, desc_dist_sum, desc_rand_dist_sum)
            for (ptrdiff_t i = 0; i < kps0.size(); ++i) {
                cv::Point2f p01 = ps01[i]; // взяли ожидаемую координату куда должна была перейти точка
                if (p01.x > 0 && p01.x < width && p01.y > 0 && p01.y < height) {
                    n_in_bounds += 1; // засчитали точку как "не вышла за пределы картинки - имеет шансы на успешное сопоставление"
                } else {
                    continue;
                }

                ptrdiff_t closest_j = -1; // будем искать ближайшую точку детектированную на искаженном изображении
                double min_error = std::numeric_limits<float>::max();
                for (ptrdiff_t j = 0; j < kps1.size(); ++j) {
                    double error = cv::norm(kps1[j].pt - p01);
                    if (error < min_error) {
                        min_error = error;
                        closest_j = j;
                    }
                }
                if (closest_j != -1 && min_error <= MAX_ACCEPTED_PIXEL_ERROR * width) {
                    // мы нашли что-то достаточно близкое - успех!
                    #pragma omp critical
                    {
                        is_not_matched0[i] = false;
                        is_not_matched1[closest_j] = false;
                    };
                    ++n_matched;
                    error_sum += min_error;
                    if (kps0[i].size != 0.0) {
                        size_ratio_sum += kps1[closest_j].size / kps0[i].size;
                    }
                    angle_diff_sum += diffAngles(kps0[i].angle, kps1[closest_j].angle);

                    cv::Mat d0 = desc0.rowRange(cv::Range(i, i + 1));
                    cv::Mat d1 = desc1.rowRange(cv::Range(closest_j, closest_j + 1));
                    size_t random_j = (239017 * i + 1232142) % kps1.size();
                    cv::Mat random_d1 = desc1.rowRange(cv::Range(random_j, random_j + 1));
                    ;
                    if (method_name == "ORB") {
                        desc_rand_dist_sum += cv::norm(d0, random_d1, cv::NORM_HAMMING);

                        desc_dist_sum += cv::norm(d0, d1, cv::NORM_HAMMING);
                    } else if (method_name == "SIFTOCV" || method_name == "SIFT_MY") {
                        desc_rand_dist_sum += cv::norm(d0, random_d1, cv::NORM_L2);

                        desc_dist_sum += cv::norm(d0, d1, cv::NORM_L2);

                        // Это способ заглянуть в черную коробку, так вы можете визуально посмотреть на то
                        // что за числа в дескрипторах двух сопоставленных точек, насколько они похожи,
                        // и сверить что расстояние между дескрипторами - это действительно расстояние
                        // между точками в пространстве высокой размерности:
#if 0
                        if (i % 100 == 0) {
                            #pragma omp critical
                            {
                                std::cout << "d0: " << d0 << std::endl;
                                std::cout << "d1: " << d1 << std::endl;
                                std::cout << "d1-d0: " << d1-d0 << std::endl;
                                cv::Mat mul;
                                cv::multiply((d1-d0), (d1-d0), mul);
                                std::cout << "(d1-d0)^2: " << mul << std::endl;
                                std::cout << "sum((d1-d0)^2): " << cv::sum(mul) << std::endl;
                                std::cout << "sqrt(sum((d1-d0)^2)): " << sqrt(cv::sum(mul)[0]) << std::endl;
                                std::cout << "norm: " << cv::norm(d0, d1, cv::NORM_L2) << std::endl;
                            }
                        }
#endif
                    }
                }
            }
            rassert(n_matched > 0, 2319241421512); // это не проверка как часть тестирования, это проверка что я не набагал и что дальше не будет деления на ноль :)
            double recall = n_matched * 1.0 / n_in_bounds;
            double avg_error = error_sum / n_matched;
            std::cout << log_prefix << n_matched << "/" << n_in_bounds << " (recall=" << recall << ") with average error=" << avg_error << std::endl;
            std::cout << log_prefix << "average size ratio between matched points: " << (size_ratio_sum / n_matched) << std::endl;
            if (angle_diff_sum != 0.0) {
                std::cout << log_prefix << "average angle difference between matched points: " << (angle_diff_sum / n_matched) << " degrees" << std::endl;
                // TODO почему SIFT менее точно угадывает средний угол отклонения? изменяется ли ситуация если выкрутить параметр ORIENTATION_VOTES_PEAK_RATIO=0.999? почему?
            }
            if (desc_dist_sum != 0.0 && desc_rand_dist_sum != 0.0) {
                std::cout << log_prefix << "average descriptor distance between matched points: " << (desc_dist_sum / n_matched) << " (random distance: " << (desc_rand_dist_sum / n_matched)
                          << ") => differentiability=" << (desc_dist_sum / desc_rand_dist_sum) << std::endl;
            }

            // а вот это проверка качества, самая важная часть теста, проверяем насколько часто одни и те же характерные точки детектируются
            // несмотря на несущественное искажение изображения
            // т.е. мы по сути проверяем что "ключевые точки детектируются инвариантно к смещению, повороту и масштабу"
            EXPECT_GT(recall, minRecall);
            // и проверяем среднюю ошибку в пикселях
            EXPECT_LT(avg_error, MAX_AVG_PIXEL_ERROR * width);

            cv::Mat result0 = img0.clone();
            cv::Mat result1 = img1.clone();
            // рисует отладочные картинки, это удобно делать по коду вообще везде, чтобы легко и удобно всегда было заглянуть в черную коробку чтобы попробовать понять
            // где проблемы, или где можно что-то улучшить
            drawKeyPoints(result0, kps0, is_not_matched0);
            drawKeyPoints(result1, kps1, is_not_matched1);

            cv::Mat result = concatenateImagesLeftRight(result0, result1);
            cv::putText(result, log_prefix + " recall=" + to_string(recall), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 0.75, CV_RGB(255, 255, 0));
            cv::putText(result, "avgPixelsError=" + to_string(avg_error), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 0.75, CV_RGB(255, 255, 0));

            // отладочную визуализацию сохраняем в папку чтобы легко было посмотреть на любой промежуточный результат
            // или в данном случае - на любой результат любого теста
            cv::imwrite("data/debug/test_sift/" + getTestSuiteName() + "/" + getTestName() + "_" + method_name + ".png", result);

            if (SHOW_RESULTS) {
                // показать результат сразу в диалоге удобно если вы запускаете один и тот же тест раз за разом
                // и хотите сразу видеть результат чтобы его оценить, вместо того чтобы идти в папочку и кликать по файлу
                cv::imshow("Red thick circles - not matched", result);
                cv::waitKey();
            }
        }
    }
}

// создаем матрицу описывающую преобразование пространства "сдвиг на вектор"
cv::Mat createTranslationMatrix(double dx, double dy)
{
    // [1, 0, dx]
    // [0, 1, dy]
    cv::Mat M = cv::Mat(2, 3, CV_64FC1, 0.0);
    M.at<double>(0, 0) = 1.0;
    M.at<double>(1, 1) = 1.0;
    M.at<double>(0, 2) = dx;
    M.at<double>(1, 2) = dy;
    return M;
}

TEST(SIFT, MovedTheSameImage)
{
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, 0.0), minRecall);
}

TEST(SIFT, MovedImageRight)
{
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(50.0, 0.0), minRecall);
}

TEST(SIFT, MovedImageLeft)
{
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(-50.0, 0.0), minRecall);
}

TEST(SIFT, MovedImageUpHalfPixel)
{
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, -50.5), minRecall);
}

TEST(SIFT, MovedImageDownHalfPixel)
{
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, 50.5), minRecall);
}

TEST(SIFT, Rotate10)
{
    double angleDegreesClockwise = 10;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate20)
{
    double angleDegreesClockwise = 20;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate30)
{
    double angleDegreesClockwise = 30;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate40)
{
    double angleDegreesClockwise = 40;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate45)
{
    double angleDegreesClockwise = 45;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate90)
{
    double angleDegreesClockwise = 90;
    double scale = 1.0;
    double minRecall = 0.75;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale50)
{
    double angleDegreesClockwise = 0;
    double scale = 0.5;
    double minRecall = 0.40;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale70)
{
    double angleDegreesClockwise = 0;
    double scale = 0.7;
    double minRecall = 0.40;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale90)
{
    double angleDegreesClockwise = 0;
    double scale = 0.9;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale110)
{
    double angleDegreesClockwise = 0;
    double scale = 1.1;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale130)
{
    double angleDegreesClockwise = 0;
    double scale = 1.3;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale150)
{
    double angleDegreesClockwise = 0;
    double scale = 1.5;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale175)
{
    double angleDegreesClockwise = 0;
    double scale = 1.75;
    double minRecall = 0.40;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Scale200)
{
    double angleDegreesClockwise = 0;
    double scale = 2.0;
    double minRecall = 0.20;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate10Scale90)
{
    double angleDegreesClockwise = 10;
    double scale = 0.9;
    double minRecall = 0.65;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, Rotate30Scale75)
{
    double angleDegreesClockwise = 30;
    double scale = 0.75;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST(SIFT, HerzJesu19RotateM40)
{
    cv::Mat jesu19 = cv::imread("data/src/test_sift/herzjesu19.png");

    ASSERT_FALSE(jesu19.empty()); // проверка что картинка была загружена
    // убедитесь что рабочая папка (Edit Configurations...->Working directory) указывает на корневую папку проекта

    double angleDegreesClockwise = -40;
    double scale = 1.0;
    double minRecall = 0.75;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(jesu19.cols / 2, jesu19.rows / 2), -angleDegreesClockwise, scale), minRecall, jesu19);
}

TEST(SIFT, DetectionSmokeTest)
{
#if ENABLE_MY_SIFT_TESTING
    phg::SIFTParams p;
    phg::SIFT sift(p, 2, "data/debug/test_sift/debug/");

    cv::Mat img = cv::imread("data/src/test_sift/mysh1.jpg");
    cv::resize(img, img, img.size() / 4, 0, 0, cv::INTER_AREA);

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    sift.detectAndCompute(img, kpts, desc);
#else
    std::cout << "ENABLE_MY_SIFT_TESTING is disabled, test skipped" << std::endl;
#endif
}

namespace fs = std::filesystem;

namespace {

const std::string kDataDir = "data/src/test_sift/test_steps_data/";
const double kRelEps = 0.05;

// ── helpers ────────────────────────────────────────────────────────────────

// Compare two matrices: same size/type, all float values within relative eps.
// Returns empty string on success, or a description of the first mismatch.
std::string compareMats(const cv::Mat& a, const cv::Mat& b, const std::string& label, double relEps)
{
    if (a.size() != b.size()) {
        std::ostringstream ss;
        ss << label << ": size mismatch " << a.size() << " vs " << b.size();
        return ss.str();
    }
    if (a.type() != b.type()) {
        std::ostringstream ss;
        ss << label << ": type mismatch " << a.type() << " vs " << b.type();
        return ss.str();
    }
    if (a.empty() && b.empty())
        return { };

    // Convert to float64 for comparison
    cv::Mat af, bf;
    a.reshape(1).convertTo(af, CV_64F);
    b.reshape(1).convertTo(bf, CV_64F);

    for (int r = 0; r < af.rows; ++r) {
        for (int c = 0; c < af.cols; ++c) {
            double va = af.at<double>(r, c);
            double vb = bf.at<double>(r, c);
            double denom = std::max({ std::abs(va), std::abs(vb), 1e-3 });
            double rel = std::abs(va - vb) / denom;
            if (rel > relEps) {
                std::ostringstream ss;
                ss << label << ": mismatch at (" << r << "," << c << "): " << va << " vs " << vb << "  (rel=" << rel << ")";
                return ss.str();
            }
        }
    }
    return { };
}

// ── Octave serialization ──────────────────────────────────────────────────

void saveOctaves(const std::string& path, const std::vector<phg::SIFT::Octave>& octaves)
{

    if (DENY_CREATE_REF_DATA)
        throw std::runtime_error("saving reference data is denied");

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "num_octaves" << (int)octaves.size();
    for (size_t i = 0; i < octaves.size(); ++i) {
        std::string prefix = "octave_" + std::to_string(i);
        fs << (prefix + "_num_layers") << (int)octaves[i].layers.size();
        for (size_t j = 0; j < octaves[i].layers.size(); ++j) {
            fs << (prefix + "_layer_" + std::to_string(j)) << octaves[i].layers[j];
        }
    }
}

std::vector<phg::SIFT::Octave> loadOctaves(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    int numOctaves;
    fs["num_octaves"] >> numOctaves;
    std::vector<phg::SIFT::Octave> octaves(numOctaves);
    for (int i = 0; i < numOctaves; ++i) {
        std::string prefix = "octave_" + std::to_string(i);
        int numLayers;
        fs[prefix + "_num_layers"] >> numLayers;
        octaves[i].layers.resize(numLayers);
        for (int j = 0; j < numLayers; ++j) {
            fs[prefix + "_layer_" + std::to_string(j)] >> octaves[i].layers[j];
        }
    }
    return octaves;
}

void compareOctaves(const std::vector<phg::SIFT::Octave>& a, const std::vector<phg::SIFT::Octave>& b, const std::string& stepName)
{
    ASSERT_EQ(a.size(), b.size()) << stepName << ": octave count mismatch";
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i].layers.size(), b[i].layers.size()) << stepName << ": octave " << i << " layer count mismatch";
        for (size_t j = 0; j < a[i].layers.size(); ++j) {
            std::string label = stepName + " oct" + std::to_string(i) + " layer" + std::to_string(j);
            std::string err = compareMats(a[i].layers[j], b[i].layers[j], label, kRelEps);
            EXPECT_TRUE(err.empty()) << err;
        }
    }
}

// ── KeyPoint serialization ────────────────────────────────────────────────

void saveKeypoints(const std::string& path, const std::vector<cv::KeyPoint>& kpts)
{
    if (DENY_CREATE_REF_DATA)
        throw std::runtime_error("saving reference data is denied");
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "keypoints" << kpts;
}

std::vector<cv::KeyPoint> loadKeypoints(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    std::vector<cv::KeyPoint> kpts;
    fs["keypoints"] >> kpts;
    return kpts;
}

// Compare two descriptor rows (as double). Returns true if all elements are
// within relative epsilon.
bool descriptorRowsSimilar(const double* a, const double* b, int cols, double relEps)
{
    double diffSq = 0.0, normASq = 0.0, normBSq = 0.0;
    for (int c = 0; c < cols; ++c) {
        double d = a[c] - b[c];
        diffSq += d * d;
        normASq += a[c] * a[c];
        normBSq += b[c] * b[c];
    }
    double denom = std::max({ std::sqrt(normASq), std::sqrt(normBSq), 1e-3 });
    return std::sqrt(diffSq) / denom <= relEps;
}

// Check if two keypoints are "similar" — all continuous fields within relative
// epsilon, and integer fields (octave, class_id) match exactly.
// If descriptor rows are provided (non-null), they must also be similar.
bool keypointsSimilar(const cv::KeyPoint& a, const cv::KeyPoint& b, double relEps, const double* descRowA = nullptr, const double* descRowB = nullptr, int descCols = 0)
{
    auto relClose = [relEps](double va, double vb) {
        double denom = std::max({ std::abs(va), std::abs(vb), 1e-3 });
        return std::abs(va - vb) / denom <= relEps;
    };
    // Angle wraps around 360, so handle the wraparound case
    auto angleDist = [](double a, double b) {
        double d = std::abs(a - b);
        return std::min(d, 360.0 - d);
    };
    if (a.octave != b.octave)
        return false;
    if (a.class_id != b.class_id)
        return false;
    if (!relClose(a.pt.x, b.pt.x))
        return false;
    if (!relClose(a.pt.y, b.pt.y))
        return false;
    if (!relClose(a.size, b.size))
        return false;
    if (!relClose(a.response, b.response))
        return false;
    // For angle: use absolute threshold (relEps * 360) to handle near-zero angles
    if (angleDist(a.angle, b.angle) > relEps * 360.0)
        return false;
    // If descriptors are provided, they must also match
    if (descRowA && descRowB && descCols > 0 && !descriptorRowsSimilar(descRowA, descRowB, descCols, 4 * relEps))
        return false;
    return true;
}

// For each detected keypoint, check that a similar keypoint exists in the
// reference set.  When descriptors are provided, similarity includes the
// descriptor row.  Test passes if:
//   1) total detected count is within 20% of reference count
//   2) at least 80% of detected keypoints have a similar reference keypoint
void compareKeypoints(const std::vector<cv::KeyPoint>& ref, const std::vector<cv::KeyPoint>& detected, const std::string& stepName, const cv::Mat& refDesc = cv::Mat(), const cv::Mat& detDesc = cv::Mat())
{
    const double kCountTolerance = 0.20; // 20%
    const double kMinMatchRate = 0.80; // 80%

    // Validate descriptor dimensions if provided
    bool useDesc = !refDesc.empty() && !detDesc.empty();
    if (useDesc) {
        ASSERT_EQ(refDesc.rows, (int)ref.size()) << stepName << ": ref descriptor row count != ref keypoint count";
        ASSERT_EQ(detDesc.rows, (int)detected.size()) << stepName << ": det descriptor row count != det keypoint count";
        ASSERT_EQ(refDesc.cols, detDesc.cols) << stepName << ": descriptor column count mismatch";
    }

    // Convert descriptors to CV_64F for uniform comparison
    cv::Mat refDescF, detDescF;
    int descCols = 0;
    if (useDesc) {
        refDesc.convertTo(refDescF, CV_64F);
        detDesc.convertTo(detDescF, CV_64F);
        descCols = refDescF.cols;
    }

    // 1) Check that counts are within 10%
    double refCount = (double)ref.size();
    double detCount = (double)detected.size();
    double countRatio = (refCount > 0) ? std::abs(detCount - refCount) / refCount : detCount;
    EXPECT_LE(countRatio, kCountTolerance) << stepName << ": keypoint count out of tolerance — detected " << detected.size() << ", reference " << ref.size() << " (diff " << (countRatio * 100.0) << "%)";

    int goodCount = 0;
    for (size_t i = 0; i < detected.size(); ++i) {
        const double* detRow = useDesc ? detDescF.ptr<double>((int)i) : nullptr;

        bool found = false;
        for (size_t j = 0; j < ref.size(); ++j) {
            const double* refRow = useDesc ? refDescF.ptr<double>((int)j) : nullptr;
            if (keypointsSimilar(detected[i], ref[j], kRelEps, detRow, refRow, descCols)) {
                found = true;
                break;
            }
        }
        if (found)
            ++goodCount;
    }

    double matchRate = (detected.empty()) ? 1.0 : (double)goodCount / detCount;
    EXPECT_GE(matchRate, kMinMatchRate) << stepName << ": only " << goodCount << " / " << detected.size() << " detected keypoints (" << (matchRate * 100.0) << "%) matched a reference keypoint (need " << (kMinMatchRate * 100.0) << "%)"
                                        << (useDesc ? " [with descriptors]" : "");

    std::cout << "[  MATCH  ] " << stepName << ": " << goodCount << "/" << detected.size() << " matched (" << (matchRate * 100.0) << "%), ref count=" << ref.size() << (useDesc ? " [with descriptors]" : "") << std::endl;
}

// ── Generic "check or create" wrappers ────────────────────────────────────

void checkOrCreateOctaves(const std::string& filename, const std::vector<phg::SIFT::Octave>& octaves, const std::string& stepName)
{
    std::string path = kDataDir + filename;
    if (fs::exists(path)) {
        auto ref = loadOctaves(path);
        compareOctaves(ref, octaves, stepName);
        std::cout << "[  CHECK  ] " << stepName << ": matched reference " << filename << std::endl;
    } else {
        saveOctaves(path, octaves);
        std::cout << "[  CREATE ] " << stepName << ": saved reference " << filename << std::endl;
    }
}

void saveDescriptors(const std::string& path, const cv::Mat& desc)
{

    if (DENY_CREATE_REF_DATA)
        throw std::runtime_error("saving reference data is denied");

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "descriptors" << desc;
}

cv::Mat loadDescriptors(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    cv::Mat desc;
    fs["descriptors"] >> desc;
    return desc;
}

// Check or create keypoints, with optional descriptors.
// When descFilename is non-empty and desc is non-empty, descriptors are
// saved/loaded alongside keypoints and included in the similarity check.
void checkOrCreateKeypoints(const std::string& filename, const std::vector<cv::KeyPoint>& kpts, const std::string& stepName, const std::string& descFilename = "", const cv::Mat& desc = cv::Mat())
{
    std::string path = kDataDir + filename;
    bool withDesc = !descFilename.empty() && !desc.empty();
    std::string descPath = withDesc ? (kDataDir + descFilename) : "";

    if (fs::exists(path)) {
        auto refKpts = loadKeypoints(path);

        cv::Mat refDesc;
        if (withDesc && fs::exists(descPath)) {
            refDesc = loadDescriptors(descPath);
        }

        compareKeypoints(refKpts, kpts, stepName, refDesc, desc);
    } else {
        saveKeypoints(path, kpts);
        if (withDesc) {
            saveDescriptors(descPath, desc);
        }
        std::cout << "[  CREATE ] " << stepName << ": saved reference " << filename << (withDesc ? " + " + descFilename : "") << std::endl;
    }
}

} // namespace

TEST(SIFT, DetectionDescriptionSteps)
{
#if ENABLE_MY_SIFT_TESTING
    ASSERT_TRUE(fs::exists(kDataDir)) << "Test data directory not found: " << kDataDir;

    phg::SIFTParams p;
    p.upscale_first = false;

    cv::Mat img = cv::imread("data/src/test_sift/mysh1.jpg");
    ASSERT_FALSE(img.empty()) << "Failed to load test image mysh1.jpg";
    cv::resize(img, img, img.size() / 8, 0, 0, cv::INTER_AREA);

    std::cout << "loaded image of size: " << img.size() << std::endl;

    cv::Mat gray = phg::toGray32F(img);

    // Step 1: Build octaves (Gaussian scale-space)
    std::vector<phg::SIFT::Octave> octaves = buildOctaves(gray, p);
    checkOrCreateOctaves("step1_octaves.yml.gz", octaves, "buildOctaves");
    if (::testing::Test::HasFatalFailure())
        return;

    // Step 2: Build Difference-of-Gaussians
    std::vector<phg::SIFT::Octave> dog = buildDoG(octaves, p);
    checkOrCreateOctaves("step2_dog.yml.gz", dog, "buildDoG");
    if (::testing::Test::HasFatalFailure())
        return;

    // Step 3: Find scale-space extrema
    std::vector<cv::KeyPoint> kpts = findScaleSpaceExtrema(dog, p);
    checkOrCreateKeypoints("step3_extrema.yml.gz", kpts, "findScaleSpaceExtrema");
    if (::testing::Test::HasFatalFailure())
        return;
    std::cout << "detected n keypoints: " << kpts.size() << std::endl;

    // Step 4: Select top keypoints (first pass)
    kpts = selectTopKeypoints(kpts, p);
    checkOrCreateKeypoints("step4_top_kpts1.yml.gz", kpts, "selectTopKeypoints_1");
    if (::testing::Test::HasFatalFailure())
        return;

    std::cout << "selected n keypoints: " << kpts.size() << std::endl;

    // Step 5: Compute orientations
    kpts = computeOrientations(kpts, octaves, p);
    checkOrCreateKeypoints("step5_orientations.yml.gz", kpts, "computeOrientations");
    if (::testing::Test::HasFatalFailure())
        return;

    std::cout << "oriented n keypoints: " << kpts.size() << std::endl;

    // Step 6: Select top keypoints (second pass)
    kpts = selectTopKeypoints(kpts, p);
    checkOrCreateKeypoints("step6_top_kpts2.yml.gz", kpts, "selectTopKeypoints_2");
    if (::testing::Test::HasFatalFailure())
        return;

    std::cout << "selected n keypoints: " << kpts.size() << std::endl;

    // Step 7: Compute descriptors
    cv::Mat desc;
    std::tie(desc, kpts) = computeDescriptors(kpts, octaves, p);
    checkOrCreateKeypoints("step7_desc_kpts.yml.gz", kpts, "computeDescriptors", "step7_descriptors.yml.gz", desc);
    if (::testing::Test::HasFatalFailure())
        return;

    std::cout << "described n keypoints: " << kpts.size() << std::endl;
#else
    std::cout << "ENABLE_MY_SIFT_TESTING is disabled, test skipped" << std::endl;
#endif
}

TEST(SIFT, PairMatching)
{
#if ENABLE_MY_SIFT_TESTING
    cv::Mat img1 = cv::imread("data/src/test_sift/mysh2.jpg");
    ASSERT_FALSE(img1.empty());

    cv::Mat img2 = cv::imread("data/src/test_sift/mysh3.jpg");
    ASSERT_FALSE(img2.empty());

    cv::resize(img1, img1, img1.size() / 2, 0, 0, cv::INTER_AREA);
    cv::resize(img2, img2, img2.size() / 2, 0, 0, cv::INTER_AREA);
    std::cout << "image sizes: " << img1.size() << ", " << img2.size() << std::endl;

    phg::SIFTParams params;
    params.nfeatures = 10000;

    std::cout << "matching using opencv orb..." << std::endl;
    auto orb_cv = cv::ORB::create(params.nfeatures);
    evaluateMatching(*orb_cv, img1, img2, "data/debug/test_sift/SIFT/Matches_ORB.jpg");

    std::cout << "matching using opencv sift..." << std::endl;
    auto sift_cv = cv::SIFT::create(params.nfeatures, params.n_octave_layers, params.contrast_threshold, params.edge_threshold);
    MatchingPairData data_cv = evaluateMatching(*sift_cv, img1, img2, "data/debug/test_sift/SIFT/Matches_SIFTOCV.jpg");

    std::cout << "matching using my sift..." << std::endl;
    phg::SIFT sift(params);
    MatchingPairData data = evaluateMatching(sift, img1, img2, "data/debug/test_sift/SIFT/Matches_SIFT_MY.jpg");

    double thresh = 0.8; // expect at least 80% of opencv sift points & matches
    EXPECT_GE(data.npoints1, thresh * data_cv.npoints1);
    EXPECT_GE(data.npoints2, thresh * data_cv.npoints2);
    EXPECT_GE(data.nmatches, thresh * data_cv.nmatches);

    std::cout << "Final score: " << data.nmatches << std::endl;
#else
    std::cout << "ENABLE_MY_SIFT_TESTING is disabled, test skipped" << std::endl;
    std::cout << "Final score: UNKNOWN" << std::endl;
#endif
}
