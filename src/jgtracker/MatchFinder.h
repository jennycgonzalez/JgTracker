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

#ifndef JG_MATCH_FINDER_H
#define JG_MATCH_FINDER_H

#include <memory>
#include "jgtracker/types.h"
#include "jgtracker/Target.h"

namespace jg {

struct PotentialMatch {
  size_t detection_index;
  std::vector<cv::DMatch> matches;
};

struct Match {
  Target* target_p = nullptr;
  Target* detection_p = nullptr;
  cv::Point2d estimated_new_centroid;
  std::vector<cv::DMatch> matches;
};

struct MatchResult {
  std::vector<Target*> unmatched_targets_p;
  std::vector<Target*> unassigned_detections_p;
  std::vector<Match> matches;
};

class MatchFinder {
 public:
  MatchFinder() = default;
  MatchFinder(MatchFinder&&) = default;
  MatchFinder& operator=(MatchFinder&&) = default;

  MatchFinder(KeypointDescriptorEnum keypoint_descriptor_enum,
              const std::string& config_file_);

  std::vector<std::vector<cv::DMatch>> ComputeAllDescriptorMatches(
      const cv::Mat& descriptors_a, const cv::Mat& descriptors_b) const;

  std::vector<cv::DMatch> ComputeDescriptorMatches(
      const cv::Mat& descriptors_a, const cv::Mat& descriptors_b) const;

  std::vector<cv::KeyPoint> ComputeDescriptorMatches(
      const std::vector<cv::KeyPoint>& keypoints_a,
      const cv::Mat& descriptors_a,
      const std::vector<cv::KeyPoint>& original_keypoints,
      const cv::Mat& original_descriptors,
      const cv::Mat& background_descriptors) const;

  std::vector<cv::DMatch> ComputeSearchAreaMatchedKeyPoints(
      const cv::Mat& descriptors_a, const cv::Mat& background_descriptors,
      const std::vector<cv::KeyPoint>& search_area_keypoints,
      const cv::Mat& search_area_descriptors,
      std::vector<cv::KeyPoint>& matched_keypoints,
      cv::Mat& new_matched_descriptors);

  static void DrawKeypointMatches(const std::vector<cv::KeyPoint>& keypoints_a,
                                  const cv::Mat& im_a,
                                  const std::vector<cv::KeyPoint>& keypoints_b,
                                  const cv::Mat& im_b,
                                  const std::vector<cv::DMatch>& matches,
                                  const std::string& image_name);

  std::vector<cv::DMatch> MatchWithTransformedOriginalKeyPoints(
      const std::vector<cv::KeyPoint>& original_conditioned_keypoints,
      const cv::Mat& original_descriptors, float scale, float rotation,
      const std::vector<cv::KeyPoint>& candidates_keypoints,
      const cv::Mat& candidates_descriptors,
      const cv::Point2f& candidates_centroid,
      std::vector<cv::KeyPoint>& matched_keypoints) const;

 private:
  std::string config_file_;
  bool cross_check_ = false;
  // From CMT paper
  double ratio_test_threshold_ = 0.4;
  float threshold_descriptor_confidence_ = 0.75;
  float threshold_descriptor_ratio_ = 0.8;
  float threshold_cutoff_ = 20;
  cv::Ptr<cv::DescriptorMatcher> matcher_p_;
};

}  // namespace jg

#endif  // JG_MATCH_FINDER_H
