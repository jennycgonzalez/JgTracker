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

#include "MatchFinder.h"

#include <cmath>
#include <utility>
#include <boost/property_tree/ini_parser.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann.hpp>
#include "jgtracker/Histogram.h"
#include "jgtracker/operations.h"
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"

namespace jg {

namespace {

const double kScaleSimilarity = 1000;
const int kMaximizeUtil = 1;
const double kMaxDistance = 15;
// const double kMaxDistance = 2; //for subway
const double kRatioTest = 0.8;
const size_t kDummyId = 1;
const cv::Scalar kDummyColor(120, 120, 120);

const int kNumNearestNeighbours = 2;
const double kRatio = 0.4;  // 0.8

// Taken from
// opencv-3.1.0/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
int RatioTest(std::vector<std::vector<cv::DMatch>> &matches, double ratio) {
  int removed = 0;
  // for all matches
  for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator =
           matches.begin();
       matchIterator != matches.end(); ++matchIterator) {
    // if 2 NN has been identified
    if (matchIterator->size() > 1) {
      // check distance ratio
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio) {
        matchIterator->clear();  // remove match
        removed++;
      }
    } else {                   // does not have 2 neighbours
      matchIterator->clear();  // remove match
      removed++;
    }
  }
  return removed;
}

std::vector<cv::DMatch> SymmetryMatches(
    const std::vector<std::vector<cv::DMatch>> &matches1,
    const std::vector<std::vector<cv::DMatch>> &matches2) {
  std::vector<cv::DMatch> symmetric_matches;

  // for all matches image 1 -> image 2
  for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 =
           matches1.begin();
       matchIterator1 != matches1.end(); ++matchIterator1) {
    // ignore deleted matches
    if (matchIterator1->empty() || matchIterator1->size() < 2) continue;

    // for all matches image 2 -> image 1
    for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 =
             matches2.begin();
         matchIterator2 != matches2.end(); ++matchIterator2) {
      // ignore deleted matches
      if (matchIterator2->empty() || matchIterator2->size() < 2) continue;

      // Match symmetry test
      if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
          (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
        // add symmetrical match
        symmetric_matches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
                                               (*matchIterator1)[0].trainIdx,
                                               (*matchIterator1)[0].distance));
        break;  // next match in image 1 -> image 2
      }
    }
  }
  return symmetric_matches;
}

int **AllocCostMatrix(std::size_t num_rows, std::size_t num_cols) {
  auto rows = static_cast<int **>(std::calloc(num_rows, sizeof(int *)));
  for (std::size_t i = 0; i != num_rows; ++i) {
    rows[i] = static_cast<int *>(std::calloc(num_cols, sizeof(int)));
  }
  return rows;
}

void FreeCostMatrix(int **matrix, std::size_t num_rows) {
  for (std::size_t i = 0; i != num_rows; ++i) {
    std::free(matrix[i]);
  }
  std::free(matrix);
}

double MinDistanceBetweenCorners(const Target &detection,
                                 const Target &target) {
  double min_distance = std::numeric_limits<double>::max();

  std::vector<cv::Point2f> detection_corners = detection.ComputeCorners();
  std::vector<cv::Point2f> target_corners = target.ComputeCorners();

  for (const cv::Point2f &det_corner : detection_corners) {
    if (target.bounding_box.contains(det_corner)) {
      for (const cv::Point2f &target_corner : target_corners) {
        min_distance =
            (-1.0) * std::min(min_distance, ComputeEuclideanDistance(
                                                target_corner, det_corner));
      }
      return min_distance;
    } else {
      for (const cv::Point2f &target_corner : target_corners) {
        min_distance = std::min(
            min_distance, ComputeEuclideanDistance(target_corner, det_corner));
      }
    }
  }
  return min_distance;
}

double ComputeSimilarity(const Target *a, const Target *b) {
  mt::check(a->HasHistogram(), "The target a does not have an histogram");
  mt::check(b->HasHistogram(), "The target b does not have an histogram");

  return a->current_histogram_p->ComputeSimilarity(
      *(b->GetOriginalHistogram()));
}

double ComputeSimilarityToOriginal(const Target *a, const Target *b) {
  mt::check(a->HasHistogram(), "The target a does not have an histogram");
  mt::check(b->HasHistogram(), "The target b does not have an histogram");

  return a->original_histogram_p->ComputeSimilarity(
      *(b->GetOriginalHistogram()));
}

void PrintSimilarityMatrix(size_t num_detections, size_t num_targets,
                           const std::vector<Target> &detections,
                           const std::vector<Target> &targets,
                           int **similarity_matrix) {
  log() << "Begin of cost matrix" << std::endl;
  log() << "| det_id:  ";
  for (size_t it = 0; it < num_detections; ++it) {
    log() << "| " << it << " ";
  }
  log() << "|" << std::endl;
  for (size_t target_it = 0; target_it < num_targets; ++target_it) {
    log() << "| targ_id:" << targets.at(target_it).id << "|";
    for (size_t it = 0; it < num_detections; ++it) {
      log() << " " << similarity_matrix[target_it][detections.at(it).id]
            << " | ";
    }
    log() << "|" << std::endl;
  }
  log() << "end of cost matrix" << std::endl;
}

}  // namespace

//------------------------------------------------------------------------------
// MatchFinder
//------------------------------------------------------------------------------

MatchFinder::MatchFinder(KeypointDescriptorEnum keypoint_descriptor_enum,
                         const std::string &config_file)
    : config_file_(config_file) {
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);

  if ((keypoint_descriptor_enum == kBRISKDescriptor) ||
      (keypoint_descriptor_enum == kORBDescriptor)) {
    matcher_p_ = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, cross_check_);
  } else {  // SIFT or SURF
    matcher_p_ = cv::makePtr<cv::BFMatcher>(cv::NORM_L2, cross_check_);
  }

  ratio_test_threshold_ =
      config.get<std::double_t>("KeypointMatch.ratio_test_threshold");

  threshold_descriptor_confidence_ =
      config.get<float>("KeypointMatch.threshold_descriptor_confidence");

  threshold_descriptor_ratio_ =
      config.get<float>("KeypointMatch.threshold_descriptor_ratio");

  threshold_cutoff_ = config.get<float>("KeypointMatch.threshold_cutoff");
}

//------------------------------------------------------------------------------
// Match Finder
//------------------------------------------------------------------------------

std::vector<cv::DMatch> MatchFinder::ComputeDescriptorMatches(
    const cv::Mat &descriptors_a, const cv::Mat &descriptors_b) const {
  std::vector<cv::DMatch> matches;

  if (!descriptors_a.empty() && !descriptors_b.empty()) {
    std::vector<std::vector<cv::DMatch>> matches12;
    // From image 1 to image 2
    matcher_p_->knnMatch(descriptors_a, descriptors_b, matches12,
                         kNumNearestNeighbours);

    std::vector<std::vector<cv::DMatch>> matches21;
    // From image 2 to image 1
    matcher_p_->knnMatch(descriptors_b, descriptors_a, matches21,
                         kNumNearestNeighbours);

    int num_removed_matches = RatioTest(matches12, ratio_test_threshold_);
    num_removed_matches += RatioTest(matches21, ratio_test_threshold_);

    matches = SymmetryMatches(matches12, matches21);
  }
  return matches;
}

// First come candidates' descriptors and keypoints
// Than the initial descriptors (target's + background)
std::vector<cv::KeyPoint> MatchFinder::ComputeDescriptorMatches(
    const std::vector<cv::KeyPoint> &keypoints_a, const cv::Mat &descriptors_a,
    const std::vector<cv::KeyPoint> &original_keypoints,
    const cv::Mat &original_descriptors,
    const cv::Mat &background_descriptors) const {
  log() << "Start ComputeDescriptorMatchesCMT \n";
  std::vector<cv::KeyPoint> matched_keypoints;

  if (!descriptors_a.empty() && !original_descriptors.empty()) {
    cv::Mat all_descriptors;
    if (background_descriptors.rows > 0) {
      cv::vconcat(original_descriptors, background_descriptors,
                  all_descriptors);
    } else {
      all_descriptors = original_descriptors;
    }

    std::vector<std::vector<cv::DMatch>> all_matches;
    // Find distances between descriptors
    matcher_p_->knnMatch(descriptors_a, all_descriptors, all_matches,
                         kNumNearestNeighbours);

    int descriptor_length = descriptors_a.cols * 8;
    MT_ASSERT_GT(descriptor_length, 0);

    for (size_t i = 0; i != keypoints_a.size(); i++) {
      std::vector<cv::DMatch> matches = all_matches[i];

      float confidence1 = 1 - (matches[0].distance / descriptor_length);
      //      float confidence2 = matches.size() > 1
      //                              ? 1 - (matches[1].distance /
      //                              descriptor_length)
      //                              : 0;
      //      float confidence2 = matches.size() > 1
      //                              ? 1 - (matches[1].distance /
      //                              descriptor_length)
      //                              : 0;
      float confidence2 = 1 - (matches[1].distance / descriptor_length);

      MT_ASSERT_NE(1 - confidence2, 0);
      // Compute distance ratio according to Lowe
      float ratio = (1 - confidence1) / (1 - confidence2);

      if (ratio < threshold_descriptor_ratio_) continue;
      if (confidence1 > threshold_descriptor_confidence_) continue;
      if (matches[0].trainIdx >= static_cast<int>(original_keypoints.size()))
        continue;

      matched_keypoints.push_back(keypoints_a.at(i));
      matched_keypoints.back().class_id =
          original_keypoints.at(matches[0].trainIdx).class_id;
    }
  }
  log() << "Finish ComputeDescriptorMatchesCMT \n";
  return matched_keypoints;
}

void MatchFinder::DrawKeypointMatches(
    const std::vector<cv::KeyPoint> &keypoints_a, const cv::Mat &im_a,
    const std::vector<cv::KeyPoint> &keypoints_b, const cv::Mat &im_b,
    const std::vector<cv::DMatch> &matches, const std::string &image_name) {
  cv::Mat img_show;
  cv::drawMatches(im_a, keypoints_a, im_b, keypoints_b, matches, img_show,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS |
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow(image_name, img_show);
}

std::vector<std::vector<cv::DMatch>> MatchFinder::ComputeAllDescriptorMatches(
    const cv::Mat &descriptors_a, const cv::Mat &descriptors_b) const {
  std::vector<std::vector<cv::DMatch>> matches;
  log() << "Start ComputeAllDescriptorMatches \n";
  if (!descriptors_a.empty() && !descriptors_b.empty()) {
    // From image 1 to image 2
    matcher_p_->knnMatch(descriptors_a, descriptors_b, matches,
                         descriptors_b.rows);
  }
  log() << "Finish ComputeAllDescriptorMatches \n";
  return matches;
}

std::vector<cv::DMatch> MatchFinder::ComputeSearchAreaMatchedKeyPoints(
    const cv::Mat &descriptors_a, const cv::Mat &background_descriptors,
    const std::vector<cv::KeyPoint> &search_area_keypoints,
    const cv::Mat &search_area_descriptors,
    std::vector<cv::KeyPoint> &matched_keypoints,
    cv::Mat &new_matched_descriptors) {
  log() << "Started matching \n";

  std::vector<cv::DMatch> final_matches;

  if (search_area_keypoints.size() > 0) {
    cv::Mat database_descriptors;
    if (background_descriptors.rows > 0) {
      cv::vconcat(descriptors_a, background_descriptors, database_descriptors);
    } else {
      database_descriptors = descriptors_a;
    }

    std::vector<cv::DMatch> matches =
        ComputeDescriptorMatches(search_area_descriptors, database_descriptors);

    log() << "Num matches: " << matches.size() << "\n";
    new_matched_descriptors =
        cv::Mat(matches.size(), search_area_descriptors.cols,
                search_area_descriptors.type());

    int index = 0;
    for (const cv::DMatch &match : matches) {
      // Take only foreground matches
      if (match.trainIdx < descriptors_a.rows) {
        //  CV_WRAP DMatch(int _queryIdx, int _trainIdx, float _distance);
        final_matches.emplace_back(index, match.trainIdx, 1);
        matched_keypoints.push_back(search_area_keypoints.at(match.queryIdx));
        matched_keypoints.back().class_id = match.trainIdx;
        search_area_descriptors.row(match.queryIdx)
            .copyTo(new_matched_descriptors.row(index++));
      }
    }
  }

  log() << "Finished matching \n";
  return final_matches;
}

std::vector<cv::DMatch> MatchFinder::MatchWithTransformedOriginalKeyPoints(
    const std::vector<cv::KeyPoint> &original_conditioned_keypoints,
    const cv::Mat &original_descriptors, float scale, float rotation,
    const std::vector<cv::KeyPoint> &candidates_keypoints,
    const cv::Mat &candidates_descriptors,
    const cv::Point2f &candidates_centroid,
    std::vector<cv::KeyPoint> &matched_keypoints) const {
  matched_keypoints.clear();

  if (!original_descriptors.empty() && !candidates_descriptors.empty()) {
    MT_ASSERT_EQ(original_conditioned_keypoints.size(),
                 static_cast<size_t>(original_descriptors.rows));
    MT_ASSERT_EQ(candidates_keypoints.size(),
                 static_cast<size_t>(candidates_descriptors.rows));

    // Transform original keypoints
    std::vector<cv::KeyPoint> transformed_original_keypoints =
        original_conditioned_keypoints;

    for (size_t i = 0; i < original_conditioned_keypoints.size(); i++) {
      transformed_original_keypoints.at(i).pt =
          scale *
              Rotate(original_conditioned_keypoints[i].pt, -rotation) +
          candidates_centroid;
    }

    // Perform local matching
    for (size_t i = 0; i < candidates_keypoints.size(); i++) {
      // Find potential original ids for matching
      std::vector<int> potential_original_ids;

      for (size_t j = 0; j < transformed_original_keypoints.size(); j++) {
        float l2norm = cv::norm(transformed_original_keypoints[j].pt -
                                candidates_keypoints.at(i).pt);
        if (l2norm < threshold_cutoff_) {
          potential_original_ids.push_back(
              transformed_original_keypoints[j].class_id);
        }
      }
      // Eliminate duplicated ids
      std::sort(potential_original_ids.begin(), potential_original_ids.end());
      auto last = std::unique(potential_original_ids.begin(),
                              potential_original_ids.end());
      potential_original_ids.erase(last, potential_original_ids.end());

      // If there are no potential ids, skip
      if (potential_original_ids.size() == 0) continue;

      // Build descriptor matrix and classes from potential indices
      cv::Mat potential_descriptors =
          cv::Mat(potential_original_ids.size(), original_descriptors.cols,
                  original_descriptors.type());

      for (size_t k = 0; k != potential_original_ids.size(); k++) {
        original_descriptors.row(potential_original_ids[k])
            .copyTo(potential_descriptors.row(k));
      }

      // Find distances between descriptors
      std::vector<std::vector<cv::DMatch>> matches;

      matcher_p_->knnMatch(candidates_descriptors.row(i), potential_descriptors,
                           matches, kNumNearestNeighbours);

      std::vector<cv::DMatch> m = matches[0];

      int descriptor_length = original_descriptors.cols * 8;
      MT_ASSERT_GT(descriptor_length, 0);
      float distance1 = m[0].distance / descriptor_length;
      float distance2 = m.size() > 1 ? m[1].distance / descriptor_length : 1;

      if (distance1 > threshold_descriptor_confidence_) continue;
      if (distance1 / distance2 > threshold_descriptor_ratio_) continue;

      matched_keypoints.push_back(candidates_keypoints.at(i));
      matched_keypoints.back().class_id =
          potential_original_ids.at(m[0].trainIdx);

    }  // end Performing matching
  }

  std::vector<cv::DMatch> matches;
  int index = 0;
  for (const cv::KeyPoint &keypoint : matched_keypoints) {
    matches.emplace_back(index++, keypoint.class_id, 0);
  }

  return matches;
}


}  // namespace jg
