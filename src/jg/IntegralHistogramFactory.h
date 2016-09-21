#ifndef JG_INTEGRAL_HISTOGRAM_FACTORY_H
#define JG_INTEGRAL_HISTOGRAM_FACTORY_H

#include "jg/types.h"

namespace jg {

class IntegralHistogramFactory {
 public:
  virtual ~IntegralHistogramFactory() = default;
  virtual void ComputeIntegralBinaryMasks(const cv::Mat &image,
                                       std::vector<cv::Mat> & /*a_mats*/,
                                       std::vector<cv::Mat> & /*b_mats*/,
                                       std::vector<cv::Mat> & /*c_mats*/) = 0;

  virtual std::vector<float> CreateHistogramVector(
      const cv::Rect2f& region, const std::vector<cv::Mat> &a_mats,
      const std::vector<cv::Mat> &b_mats,
      const std::vector<cv::Mat> &c_mats) = 0;
};

class GrayscaleIntegralHistogramFactory : public IntegralHistogramFactory {
 public:
  void ComputeIntegralBinaryMasks(const cv::Mat &image,
                               std::vector<cv::Mat> & grayscale_mats,
                               std::vector<cv::Mat> & /*b_mats*/,
                               std::vector<cv::Mat> & /*c_mats*/) override;

  std::vector<float> CreateHistogramVector(
      const cv::Rect2f& region, const std::vector<cv::Mat> &grayscale_mats,
      const std::vector<cv::Mat> & /*b_mats*/,
      const std::vector<cv::Mat> & /*c_mats*/) override;
};

class HsvIntegralHistogramFactory : public IntegralHistogramFactory {
 public:
  void ComputeIntegralBinaryMasks(const cv::Mat &image,
                               std::vector<cv::Mat> & h_mats,
                               std::vector<cv::Mat> & s_mats,
                               std::vector<cv::Mat> & /*c_mats*/) override;

 std::vector<float> CreateHistogramVector(
      const cv::Rect2f& region, const std::vector<cv::Mat> &h_mats,
      const std::vector<cv::Mat> & s_mats,
      const std::vector<cv::Mat> & /*c_mats*/) override;
};

}  // namespace jg

#endif  // JG_INTEGRAL_HISTOGRAM_FACTORY_H
