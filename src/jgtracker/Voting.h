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

#ifndef JG_VOTING_H
#define JG_VOTING_H

#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace jg {

class Voting {
 public:
  Voting() : threshold_outlier_(20) {}

  cv::Point2f ComputeNewCentroid(
      const std::vector<cv::KeyPoint>& original_conditioned_keypoints,
      float scale, float rotation,
      const std::vector<cv::KeyPoint>& final_keypoints);

  std::vector<cv::KeyPoint> DissambiguateCandidateKeypoints(
      const std::vector<cv::KeyPoint>& candidate_keypoints,
      const std::vector<std::vector<cv::DMatch>>& matches_to_original_keypoints,
      int descriptor_lenght, const cv::Point2f& centroid,
      const std::vector<cv::KeyPoint>& trans_original_conditioned_keypoints);

  void InitializeConsensusData(const std::vector<cv::KeyPoint>& keypoints,
                               cv::Mat& distances_pairwise,
                               cv::Mat& angles_pairwise);

  void EstimateScaleRotation(
      const std::vector<cv::KeyPoint>& candidate_keypoints,
      const cv::Mat& original_distances_pairwise,
      const cv::Mat& original_angles_pairwise, float& scale, float& rotation);

  std::vector<cv::KeyPoint> FindConsensus(
      const std::vector<cv::KeyPoint>& candidates_keypoints,
      const std::vector<cv::KeyPoint>& original_conditioned_keypoints,
      float scale, float rotation, cv::Point2f& center);

 private:
  float threshold_outlier_;
  std::vector<cv::Point2f> conditioned_points_;
};

}  // namespace jg

#endif  // JG_VOTING_H
