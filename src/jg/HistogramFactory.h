#ifndef JG_HISTOGRAM_FACTORY_H
#define JG_HISTOGRAM_FACTORY_H

#include "jg/types.h"

namespace jg {

class HistogramFactory {
 public:
  virtual ~HistogramFactory() = default;

  virtual void Create(const Mat3Uchar& image, const cv::Mat& roi_mask,
                      cv::Mat& hist) = 0;
};

class GrayscaleHistogramFactory : public HistogramFactory {
 public:
  void Create(const Mat3Uchar& image, const cv::Mat& roi_mask,
              cv::Mat& hist) override;
};

class HsvHistogramFactory : public HistogramFactory {
 public:
  void Create(const Mat3Uchar& image, const cv::Mat& roi_mask,
              cv::Mat& hist) override;
};

}  // namespace jg

#endif  // JG_HISTOGRAM_FACTORY_H
