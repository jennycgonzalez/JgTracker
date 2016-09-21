#include "jg/HistogramFactory.h"

#include "jg/operations.h"
#include "jg/types.h"

namespace jg {

void HsvHistogramFactory::Create(const Mat3Uchar& image,
                                 const cv::Mat& roi_mask, cv::Mat& histogram) {
  ComputeHsvHistogram(image, roi_mask, histogram);
}

void GrayscaleHistogramFactory::Create(const Mat3Uchar& image,
                                       const cv::Mat& roi_mask,
                                       cv::Mat& histogram) {
  ComputeGrayscaleHistogram(image, roi_mask, histogram);
}

}  // namespace jg
