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

#include "Voting.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <opencv2/core/types.hpp>
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/cppmt/fastcluster/fastcluster.h"
#include "jgtracker/operations.h"
#include "jgtracker/types.h"

// Based on:
// https://github.com/gnebehay/VOTR/tree/master/CppMT
// Adapted for this project

namespace jg {

namespace {

const float kThresholdRatio = 0.8;
const float kThresholdConfidence = 0.75;

}  // namespace

cv::Point2f Voting::ComputeNewCentroid(
    const std::vector<cv::KeyPoint>& original_conditioned_keypoints,
    float scale, float rotation,
    const std::vector<cv::KeyPoint>& final_keypoints) {
  cv::Point2f centroid;
  std::vector<cv::Point2f> votes;
  cv::Point2f sum_votes(0, 0);

  for (const cv::KeyPoint& final_k : final_keypoints) {
    sum_votes +=
        final_k.pt -
        scale * Rotate(original_conditioned_keypoints.at(final_k.class_id).pt,
                       rotation);
  }

  size_t num_votes = final_keypoints.size();
  if (num_votes > 0) {
    centroid.x = sum_votes.x / num_votes;
    centroid.y = sum_votes.y / num_votes;
  } else {
    centroid.x = std::numeric_limits<float>::quiet_NaN();
    centroid.y = std::numeric_limits<float>::quiet_NaN();
  }

  return centroid;
}

void Voting::InitializeConsensusData(const std::vector<cv::KeyPoint>& keypoints,
                                     cv::Mat& distances_pairwise,
                                     cv::Mat& angles_pairwise) {
  size_t num_points = keypoints.size();

  if (num_points > 1) {
    log() << "Started creating initial distance and angle matrices \n";
    // Create matrices of pairwise distances/angles
    distances_pairwise = cv::Mat::ones(num_points, num_points, CV_32FC1);
    angles_pairwise = cv::Mat::zeros(num_points, num_points, CV_32FC1);

    for (size_t i = 0; i < num_points; i++) {
      for (size_t j = i + 1; j < num_points; j++) {
        cv::Point2f v = keypoints[i].pt - keypoints[j].pt;

        float distance = std::hypot(v.x, v.y);
        float angle = std::atan2(v.y, v.x);

        MT_ASSERT_GT(distance, 0);

        distances_pairwise.at<float>(i, j) = distance;
        distances_pairwise.at<float>(j, i) = distance;
        angles_pairwise.at<float>(i, j) = angle;
        angles_pairwise.at<float>(j, i) = angle;
      }
    }
  }
}

std::vector<cv::KeyPoint> Voting::DissambiguateCandidateKeypoints(
    const std::vector<cv::KeyPoint>& candidate_keypoints,
    const std::vector<std::vector<cv::DMatch>>& matches_to_original_keypoints,
    int descriptor_lenght, const cv::Point2f& centroid,
    const std::vector<cv::KeyPoint>& trans_original_conditioned_keypoints) {
  log() << "Start DissambiguateCandidateKeypoints \n";

  std::vector<cv::KeyPoint> final_candidates;

  size_t num_original_keypoints = trans_original_conditioned_keypoints.size();

  if (!candidate_keypoints.empty() && (num_original_keypoints > 1)) {
    for (size_t i = 0; i < candidate_keypoints.size(); i++) {
      std::vector<cv::DMatch> matches = matches_to_original_keypoints.at(i);
      MT_ASSERT_EQ(matches.size(), trans_original_conditioned_keypoints.size());
      std::sort(matches.begin(), matches.end(),
                [](const cv::DMatch& a, const cv::DMatch& b) {
                  return a.trainIdx < b.trainIdx;
                });
      std::vector<float> confidences;
      float descriptor_lenght_f = static_cast<float>(descriptor_lenght);
      for (const cv::DMatch& match : matches) {
        confidences.push_back(1 - match.distance / descriptor_lenght_f);
      }
      cv::Point2f relative_position = candidate_keypoints.at(i).pt - centroid;

      std::vector<std::pair<int, float>> combined;

      for (size_t k = 0; k != num_original_keypoints; k++) {
        cv::Point2f diff =
            relative_position - trans_original_conditioned_keypoints.at(k).pt;
        float displacement = std::hypotf(diff.x, diff.y);

        int id = trans_original_conditioned_keypoints.at(k).class_id;
        if (displacement < threshold_outlier_) {
          combined.emplace_back(id, confidences.at(k));
        } else {
          combined.emplace_back(id, 0);
        }
      }
      // Sort in descending order
      std::sort(
          combined.begin(), combined.end(),
          [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
          });

      if (combined.at(1).second == 1) {
        combined.at(1).second = 0;
      }
      float ratio = (1 - combined.at(0).second) / (1 - combined.at(1).second);

      int best_id = combined.at(0).first;
      if ((ratio < kThresholdRatio) &&
          (combined.at(0).second > kThresholdConfidence)) {
        final_candidates.push_back(candidate_keypoints.at(i));
        final_candidates.back().class_id = best_id;
      }
    }
  }
  log() << "Finish DissambiguateCandidateKeypoints \n";
  return final_candidates;
}

// TODO: Check for estimate_scale, estimate_rotation
void Voting::EstimateScaleRotation(
    const std::vector<cv::KeyPoint>& candidate_keypoints,
    const cv::Mat& original_distances_pairwise,
    const cv::Mat& original_angles_pairwise, float& scale, float& rotation) {
  log() << "Consensus::estimateScaleRotation() call \n";

  size_t num_candidates = candidate_keypoints.size();
  if (num_candidates > 1) {
    // Compute pairwise changes in scale/rotation
    std::vector<float> changes_scale;
    changes_scale.reserve(num_candidates * (num_candidates - 1));
    std::vector<float> changes_angles;
    changes_angles.reserve(num_candidates * (num_candidates - 1));

    for (size_t i = 0; i < num_candidates; i++) {
      for (size_t j = i + 1; j < num_candidates; j++) {
        cv::Point2f v =
            candidate_keypoints.at(i).pt - candidate_keypoints.at(j).pt;

        float distance = std::hypot(v.x, v.y);
        size_t row = candidate_keypoints.at(i).class_id;
        MT_ASSERT_LE(row,
                     static_cast<size_t>(original_distances_pairwise.rows));
        size_t col = candidate_keypoints.at(j).class_id;
        MT_ASSERT_LE(col,
                     static_cast<size_t>(original_distances_pairwise.cols));
        MT_ASSERT_FALSE(row == col);

        float distance_original =
            original_distances_pairwise.at<float>(row, col);
        // There must be no duplicated coordinates in the original set
        MT_ASSERT_GT(distance_original, 0);
        float change_scale = distance / distance_original;
        changes_scale.push_back(change_scale);

        float angle = std::atan2(v.y, v.x);
        float angle_original = original_angles_pairwise.at<float>(row, col);
        float change_angle = angle - angle_original;

        // Fix long way angles
        if (std::fabs(change_angle) > M_PI) {
          change_angle = change_angle > 0 ? 2 * M_PI + change_angle
                                          : -2 * M_PI + change_angle;
        }
        changes_angles.push_back(change_angle);
      }
    }

    // Do not use changes_scale, changes_angle after this point as their order
    // is changed by GetMedian()
    if (changes_scale.size() < 2) {
      scale = std::numeric_limits<float>::quiet_NaN();
    } else {
      scale = GetMedian(changes_scale);
    }

    if (changes_angles.size() < 2) {
      rotation = std::numeric_limits<float>::quiet_NaN();
    } else {
      rotation = GetMedian(changes_angles);
    }
  } else {
    scale = std::numeric_limits<float>::quiet_NaN();
    rotation = std::numeric_limits<float>::quiet_NaN();
  }
  log() << "Consensus::estimateScaleRotation() return \n";
}

std::vector<cv::KeyPoint> Voting::FindConsensus(
    const std::vector<cv::KeyPoint>& candidates_keypoints,
    const std::vector<cv::KeyPoint>& original_conditioned_keypoints,
    float scale, float rotation, cv::Point2f& center) {
  log() << "Consensus::findConsensus() call  ";

  std::vector<cv::KeyPoint> inliers;

  // If no candidates are available or no inliers are found, return nan
  center.x = std::numeric_limits<float>::quiet_NaN();
  center.y = std::numeric_limits<float>::quiet_NaN();

  if (candidates_keypoints.size() == 0) {
    log() << "Candidates Keypoints Size == 0 \n"
             "Consensus::findConsensus() return  \n";
    return inliers;
  }

  if (std::isnan(scale)) {
    log() << "Scale is not a number (nan) \n"
             "Consensus::findConsensus() return  \n";
    return inliers;
  }

  if (std::isnan(rotation)) rotation = 0;

  log() << "Consensus: compute votes step 1 \n";
  // Compute all votes including outliers
  std::vector<cv::Point2f> votes;
  for (size_t i = 0; i < candidates_keypoints.size(); i++) {
    votes.push_back(
        candidates_keypoints.at(i).pt -
        scale *
            Rotate(original_conditioned_keypoints.at(candidates_keypoints.at(i)
                                                         .class_id).pt,
                   rotation));
  }

  log() << "Consensus: compute votes step 2  \n";
  t_index N = votes.size();

  // This is a lot of memory, so we put it on the heap
  float* D = new float[N * (N - 1) / 2];

  cluster_result Z(N);  // Previous: cluster_result Z(N - 1)
                        // but this generates a bug when N = 1

  log() << "Consensus: compute votes step 3 \n";
  // Compute pairwise distances between votes
  int index = 0;
  for (size_t i = 0; i != candidates_keypoints.size() - 1; i++) {
    for (size_t j = i + 1; j != candidates_keypoints.size(); j++) {
      // TODO: This index calculation is correct, but is it a good thing?
      // int index = i * (points.size() - 1) - (i*i + i) / 2 + j - 1;
      cv::Point2f diff = votes[i] - votes[j];
      D[index] = std::hypot(diff.x, diff.y);
      index++;
    }
  }

  log() << "Consensus::MST_linkage_core() call  \n";
  MST_linkage_core(N, D, Z);
  log() << "Consensus::MST_linkage_core() return  \n";

  union_find nodes(N);

  // Sort linkage by distance ascending
  std::stable_sort(Z[0], Z[N - 1]);

  // S are cluster sizes
  int* S = new int[2 * N - 1];
  // TODO: Why does this loop go to 2*N-1? Shouldn't it be simply N? Everything
  // > N gets overwritten later
  for (int i = 0; i < 2 * N - 1; i++) {
    S[i] = 1;
  }

  // After the loop ends, parent contains the index of the last cluster
  t_index parent = 0;
  for (node const* NN = Z[0]; NN != Z[N - 1]; ++NN) {
    // Get two data points whose clusters are merged in step i.
    // Find the cluster identifiers for these points.
    t_index node1 = nodes.Find(NN->node1);
    t_index node2 = nodes.Find(NN->node2);

    // Merge the nodes in the union-find data structure by making them
    // children of a new node
    // if the distance is appropriate
    if (NN->dist < threshold_outlier_) {
      parent = nodes.Union(node1, node2);
      S[parent] = S[node1] + S[node2];
    }
  }

  // Get cluster labels
  int* T = new int[N];
  for (t_index i = 0; i < N; i++) {
    T[i] = nodes.Find(i);
  }

  // Find largest cluster
  int S_max = std::distance(S, std::max_element(S, S + 2 * N - 1));

  // Find inliers, compute center of votes
  center.x = center.y = 0;

  for (size_t i = 0; i < votes.size(); i++) {
    // If point is in consensus cluster
    if (T[i] == S_max) {
      inliers.push_back(candidates_keypoints.at(i));
      center.x += votes[i].x;
      center.y += votes[i].y;
    }
  }

  if (inliers.size() > 0) {
    center.x /= inliers.size();
    center.y /= inliers.size();
  } else {
    center.x = std::numeric_limits<float>::quiet_NaN();
    center.y = std::numeric_limits<float>::quiet_NaN();
  }

  delete[] D;
  delete[] S;
  delete[] T;

  log() << "Consensus::findConsensus() return \n";
  return inliers;
}

}  // namespace jg
