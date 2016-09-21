#ifndef JG_MATCH_FINDER_H
#define JG_MATCH_FINDER_H

#include <memory>
#include "jg/types.h"
#include "jg/Target.h"

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
