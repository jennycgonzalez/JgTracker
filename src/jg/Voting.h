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
