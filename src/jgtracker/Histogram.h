#ifndef JG_HISTOGRAM_H
#define JG_HISTOGRAM_H

#include <opencv2/core/types.hpp>
#include "jgtracker/types.h"

namespace jg {

class Histogram {
 public:
  virtual ~Histogram() = default;
  double ComputeSimilarity(const Histogram& histogram) const;
  virtual void Update(const cv::Mat& image, const MatUchar& roi_mask) = 0;
  const cv::Mat& histogram_matrix() const { return histogram_; }

 protected:
  cv::Mat histogram_;
};

class GrayscaleHistogram : public Histogram {
 public:
  void Update(const cv::Mat& image, const MatUchar& roi_mask) override;
};

class HsvHistogram : public Histogram {
 public:
  void Update(const cv::Mat& image, const MatUchar& roi_mask) override;
};

}  // namespace jg

#endif  // JG_HISTOGRAM_H
