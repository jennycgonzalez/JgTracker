#include "jgtracker/Histogram.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "jgtracker/operations.h"

namespace jg {

//------------------------------------------------------------------------------
// Histogram
//------------------------------------------------------------------------------

double Histogram::ComputeSimilarity(const Histogram& histogram) const {
  return 1.0 - cv::compareHist(histogram_, histogram.histogram_matrix(),
                               cv::HISTCMP_HELLINGER);
}

//------------------------------------------------------------------------------
// GrayscaleHistogram
//------------------------------------------------------------------------------

void GrayscaleHistogram::Update(const cv::Mat &image,
                                const MatUchar& roi_mask) {
  ComputeGrayscaleHistogram(image, roi_mask, histogram_);
  cv::normalize(histogram_, histogram_, 1, 0, cv::NORM_L1);
}

//------------------------------------------------------------------------------
// ColorHistogram
//------------------------------------------------------------------------------

void HsvHistogram::Update(const cv::Mat& image, const MatUchar& roi_mask) {
  ComputeHsvHistogram(image, roi_mask, histogram_);
  cv::normalize(histogram_, histogram_, 1, 0, cv::NORM_L1);
}

}  // namespace jg
