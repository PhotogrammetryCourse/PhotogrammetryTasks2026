#pragma once

#include <opencv2/core.hpp>

namespace phg {

struct SIFTParams {
    int nfeatures = 0;
    int n_octave_layers = 3;
    double contrast_threshold = 0.04;
    double edge_threshold = 15;
    double sigma = 1.6;

    double orient_peak_ratio = 0.84;
    int orient_nbins = 36;
    bool upscale_first = true;

    bool enable_subpixel_localization = true;
    bool enable_edge_like_filtering = true;
    bool enable_orientation_bin_interpolation = true;
    bool enable_orientation_gaussian_weighting = true;
    bool enable_orientation_subpixel_localization = true;
    bool enable_descriptor_gaussian_weighting = true;
    bool enable_descriptor_bin_interpolation = true;
};

class SIFT {
public:
    struct Octave {
        std::vector<cv::Mat> layers;
    };

    explicit SIFT(const SIFTParams& p = SIFTParams(), int verbose_level = 0, const std::string& debug_folder = "")
        : p(p)
        , verbose_level(verbose_level)
        , debug_folder(debug_folder)
    {
    }

    void detectAndCompute(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) const;
    void detectAndCompute(const cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) const { detectAndCompute(img, { }, kpts, desc); }

private:
    void saveImg(const std::string& name, const cv::Mat& img) const;
    void savePyramid(const std::string& name, const std::vector<Octave>& pyramid, bool normalize = false) const;

    int verbose_level;
    std::string debug_folder;
    SIFTParams p;
};

cv::Mat toGray32F(const cv::Mat& img);
std::vector<SIFT::Octave> buildOctaves(const cv::Mat& img, const SIFTParams& p, int verbose_level = 0);
std::vector<SIFT::Octave> buildDoG(const std::vector<SIFT::Octave>& octaves, const SIFTParams& p, int verbose_level = 0);
std::vector<cv::KeyPoint> findScaleSpaceExtrema(const std::vector<SIFT::Octave>& dog, const SIFTParams& params, int verbose_level = 0);
std::vector<cv::KeyPoint> computeOrientations(const std::vector<cv::KeyPoint>& kpts, const std::vector<SIFT::Octave>& octaves, const SIFTParams& params, int verbose_level = 0);
std::pair<cv::Mat, std::vector<cv::KeyPoint>> computeDescriptors(const std::vector<cv::KeyPoint>& kpts, const std::vector<SIFT::Octave>& octaves, const SIFTParams& params, int verbose_level = 0);
std::vector<cv::KeyPoint> selectTopKeypoints(const std::vector<cv::KeyPoint>& kpts, const SIFTParams& params, int verbose_level = 0);
}
