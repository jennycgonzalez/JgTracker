#ifndef JG_TARGET_H
#define JG_TARGET_H

#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "jgtracker/Histogram.h"

namespace jg {

struct Target {
  size_t id = 0;
  size_t age = 1;
  cv::Scalar label_color;
  cv::Rect2f bounding_box;
  cv::Size initial_size;

  size_t total_visible_counts = 1;
  size_t consecutive_invisible_count = 0;

  size_t descriptor_lenght;

  std::vector<cv::KeyPoint> active_keypoints;

  cv::Mat original_image;
  cv::Mat original_descriptors;
  std::vector<cv::KeyPoint> original_keypoints;
  std::vector<cv::KeyPoint> original_conditioned_keypoints;

  // Consensus data
  cv::Mat original_distances_pairwise;
  cv::Mat original_angles_pairwise;

  std::unique_ptr<Histogram> original_histogram_p;
  std::unique_ptr<Histogram> current_histogram_p;
  std::vector<std::vector<float>> stripes_histograms;


  Target() = default;
  Target(Target&&) = default;
  Target& operator=(Target&&) = default;

  Target(size_t _id, const cv::Scalar& color, const cv::Rect2f &box);

  void ChangeBoundingBox(const cv::Rect2f &new_bounding_box);

  const Histogram* GetOriginalHistogram() const {
    return original_histogram_p.get();
  }

  const Histogram* GetCurrentlHistogram() const {
    return current_histogram_p.get();
  }

  bool HasHistogram() const { return original_histogram_p.get(); }
  void Translate(const cv::Point2f &new_center, const cv::Rect2f&);
  void Resize(float width, float height);

  void SaveOriginalImage(const Mat3Uchar& image);
  void InitHistograms(ColorSpaceEnum color_space_enum, const cv::Mat& image);
  void UpdateCurrentHistogram(const Mat3Uchar& image);

  std::vector<cv::Point2f> ComputeCorners() const;
  const cv::Size2i GetSize() const;

  void PrintInImage(Mat3Uchar& image) const;
  void PrintInImage(Mat3Uchar& image, const cv::Scalar& color) const;
};

}  // namespace jg

#endif  // JG_TARGET_H
