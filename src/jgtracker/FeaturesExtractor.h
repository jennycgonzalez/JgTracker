//     ___     _____              _
//    |_  |   |_   _|            | |
//      | | __ _| |_ __ __ _  ___| | _____ _ __
//      | |/ _` | | '__/ _` |/ __| |/ / _ \ '__|
//  /\__/ / (_| | | | | (_| | (__|   <  __/ |
//  \____/ \__, \_/_|  \__,_|\___|_|\_\___|_|
//         __/ |
//        |___/
//
// https://github.com/jennycgonzalez/jgtracker
//
// BSD 2-Clause License

/*
Copyright (c) 2016, Jenny Gonzalez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef JG_FEATURES_EXTRACTOR_H
#define JG_FEATURES_EXTRACTOR_H

#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/tracking/feature.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "jgtracker/types.h"

namespace jg {

class FeaturesExtractor {
 public:
  FeaturesExtractor() = default;
  FeaturesExtractor(FeaturesExtractor&&) = default;
  FeaturesExtractor& operator=(FeaturesExtractor&&) = default;

  FeaturesExtractor(KeypointDetectorEnum keypoint_detector_enum,
                    KeypointDescriptorEnum keypoint_descriptor_enum,
                    const std::string& config_file);

  void UpdateFeatures(const Mat3Uchar& image, const cv::Rect2d& bounding_box,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::Mat& descriptors);

  void UpdateFeatures(const Mat3Uchar& image, const cv::Mat& roi_mask,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::Mat& descriptors);

  int GetDescriptorLenghtInBits() { return 8 * descriptor_lenght_; }

 protected:
  cv::Ptr<cv::FeatureDetector> detector_p_;
  cv::Ptr<cv::DescriptorExtractor> descriptor_p_;
  int descriptor_lenght_;
};

}  // namespace jg

#endif  // JG_FEATURES_EXTRACTOR_H
