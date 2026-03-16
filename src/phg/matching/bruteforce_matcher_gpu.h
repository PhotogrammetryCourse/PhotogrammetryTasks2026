#pragma once

#include "descriptor_matcher.h"

namespace phg {

    struct BruteforceMatcherGPU : DescriptorMatcher {

        static bool isAvailable(std::string *reason = nullptr, bool require_non_cpu_device = false);

        void train(const cv::Mat &train_desc) override;

        void knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const override;

    private:

        const cv::Mat *train_desc_ptr = nullptr;
    };

}