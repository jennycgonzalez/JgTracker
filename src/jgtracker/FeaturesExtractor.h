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
