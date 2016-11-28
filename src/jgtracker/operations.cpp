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
Copyright (c) 2016, Jenny González
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


#include "jgtracker/operations.h"

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"

namespace jg {

void AssignUniqueIdsToKeypoints(std::vector<cv::KeyPoint> &keypoints) {
  // We start the keypoints ids with 0 to keep them synchronized
  // with the descriptors rows indices
  int id = 0;
  for (cv::KeyPoint &keypoint : keypoints) {
    keypoint.class_id = id++;
  }
}

// The histogram computation was taken from:
// http://docs.opencv.org/3.1.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d&gsc.tab=0
void ComputeHsvHistogram(const Mat3Uchar &image, const cv::Mat &roi_mask,
                         cv::Mat &hist) {
  MT_ASSERT_EQ(roi_mask.type(), CV_8UC1);
  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

  // Quantize the hue to 30 levels and the saturation to 32 levels
  int hbins = 30, sbins = 32;
  int histSize[] = {hbins, sbins};
  // hue varies from 0 to 180, see cvtColor
  float hranges[] = {0, 180};  // [0, 180)
  // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
  float sranges[] = {0, 256};  // [0, 256)
  const float *ranges[] = {hranges, sranges};

  int channels[] = {0, 1};
  /// Get the Histogram and normalize it
  cv::calcHist(&hsv_image, 1, channels, roi_mask, hist, 2, histSize, ranges,
               true,  // the histogram is uniform
               false);
}

void ComputeGrayscaleHistogram(const Mat3Uchar &image, const MatUchar &roi_mask,
                               cv::Mat &hist) {
  MT_ASSERT_EQ(roi_mask.type(), CV_8UC1);
  cv::Mat grayscale_image;
  cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);

  int bins = 32;
  int histSize[] = {bins};
  float gray_ranges[] = {0, 256};  // [0, 256)
  const float *ranges[] = {gray_ranges};

  int channels[] = {0};
  /// Get the Histogram and normalize it
  cv::calcHist(&grayscale_image, 1, channels, roi_mask, hist, 1, histSize,
               ranges,
               true,  // the histogram is uniform
               false);
}

cv::Point2f ComputeCentroid(const cv::Rect2f &box) {
  float x = std::floor(box.width * 0.5) + box.x;
  float y = std::floor(box.height * 0.5) + box.y;
  return cv::Point2f(x, y);
}

void ComputeCorners(const cv::RotatedRect &box,
                    std::vector<cv::Point2f> &corners) {
  cv::Point2f vertices[4];
  box.points(vertices);
  for (size_t i = 0; i < 4; i++) {
    corners.at(i) = cv::Point2d(vertices[i]);
  }
}

double ComputeEuclideanDistance(cv::Point2d a, cv::Point2d b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

std::vector<cv::KeyPoint> ConditionKeypoints(
    const cv::Point2f &centroid, const std::vector<cv::KeyPoint> &keypoints) {
  log() << "Start conditioning keypoints \n";

  std::vector<cv::KeyPoint> conditioned_keypoints = keypoints;
  for (cv::KeyPoint &keypoint : conditioned_keypoints) {
    keypoint.pt -= centroid;
  }
  log() << "Finish conditioning keypoints \n";
  return conditioned_keypoints;
}

void ConditionPoints(const cv::Point2f &centroid,
                     const std::vector<cv::KeyPoint> &keypoints,
                     std::vector<cv::Point2f> &conditioned_points) {
  cv::KeyPoint::convert(keypoints, conditioned_points);
  for (cv::Point2f &normalized_point : conditioned_points) {
    normalized_point -= centroid;
  }
}

std::vector<cv::Rect2f> CreateImageStripes(float width, float height,
                                           size_t num_patches) {
  size_t patch_width =
      static_cast<size_t>(std::floor(width / static_cast<float>(num_patches)));
  size_t patch_height =
      static_cast<size_t>(std::floor(height / static_cast<float>(num_patches)));

  size_t const_un = 1;

  patch_width = std::max(const_un, patch_width);
  patch_height = std::max(const_un, patch_height);

  std::vector<cv::Rect2f> patches;

  // horizontal patches

  for (size_t i = 0; i < num_patches; i++) {
    cv::Point2f tl(0, static_cast<float>(i * patch_height));
    cv::Size2f patch_size(width, patch_height + 1);
    patches.emplace_back(tl, patch_size);
  }

  if (patches.back().br().y != height) {
    patches.back() = cv::Rect2f(patches.back().tl(),
                                cv::Point2f(patches.back().br().x, height));
  }

  // vertical patches
  for (size_t i = 0; i < num_patches; i++) {
    cv::Point2f tl(static_cast<float>(i * patch_width), 0);
    cv::Size2f patch_size(patch_width, height);
    patches.emplace_back(tl, patch_size);
  }

  if (patches.back().br().x != width) {
    patches.back() = cv::Rect2f(patches.back().tl(),
                                cv::Point2f(width, patches.back().br().y));
  }

  return patches;
}

void DrawKeypoints(const std::vector<cv::KeyPoint> &keypoints,
                   const Mat3Uchar &image, const cv::Scalar &color,
                   const std::string &window_name) {
  cv::Mat output;
  cv::drawKeypoints(image, keypoints, output, color);
  cv::imshow(window_name, output);
}

void EliminateDuplicatedIDKeypoints(std::vector<cv::KeyPoint> &keypoints) {
  auto are_equal = [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
    return a.class_id == b.class_id;
  };
  auto last = std::unique(keypoints.begin(), keypoints.end(), are_equal);
  keypoints.erase(last, keypoints.end());
}

float GetMedian(std::vector<float> &list) {
  if (list.size() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  std::nth_element(list.begin(), list.begin() + list.size() / 2, list.end());

  return list[list.size() / 2];
}

cv::Mat ReadTransitionMatrixFromFile(const boost::filesystem::path &filename) {
  std::ifstream stream(filename.string());
  mt::check(stream.is_open(), "Could not open '%s'", filename.c_str());

  const auto getNextRow = [&]() {
    std::string line;
    std::getline(stream, line);
    mt::check(static_cast<bool>(stream), "Fetching line from '%s' failed", filename.c_str());

    double value;
    std::vector<double> values;
    std::istringstream iss(line);
    while (iss >> value) {
      values.push_back(value);
    }
    return values;
  };

  const auto first_row = getNextRow();
  const auto dimension = first_row.size();
  MatDouble matrix = cv::Mat(dimension, dimension, CV_64FC1);

  for (size_t col = 0; col != dimension; col++) {
    matrix.at<double>(0, col) = first_row[col];
  }

  for (size_t row = 1; row != dimension; row++) {
    const auto new_row = getNextRow();
    MT_ASSERT_EQ(dimension, new_row.size());
    for (size_t col = 0; col != dimension; col++) {
      matrix.at<double>(row, col) = new_row[col];
    }
  }
  return matrix;
}

cv::Point2f Rotate(const cv::Point2f &v, float angle) {
  cv::Point2f r;
  r.x = std::cos(angle) * v.x - std::sin(angle) * v.y;
  r.y = std::sin(angle) * v.x + std::cos(angle) * v.y;
  return r;
}

std::vector<cv::KeyPoint> RotateAndScaleKeypoints(
    const std::vector<cv::KeyPoint> &keypoints, float angle, float scale) {
  std::vector<cv::KeyPoint> transformed_keypoints = keypoints;
  log() << "Start RotateAndScaleKeypoints \n";

  for (cv::KeyPoint &keypoint : transformed_keypoints) {
    keypoint.pt = scale * Rotate(keypoint.pt, angle);
  }
  log() << "Finish RotateAndScaleKeypoints \n";
  return transformed_keypoints;
}

void ShowImage(const cv::Mat &frame, const std::string &name, int corner_x,
               int corner_y) {
  cv::imshow(name, frame);
  cv::moveWindow(name, corner_x, corner_y);
}

void SortKeypointsByID(std::vector<cv::KeyPoint> &keypoints) {
  auto second_is_greater = [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
    return a.class_id < b.class_id;
  };
  std::sort(keypoints.begin(), keypoints.end(), second_is_greater);
}

MatUchar ToGrayscaleFrame(const Mat3Uchar &frame) {
  MatUchar gray_frame;
  cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
  return gray_frame;
}

std::vector<cv::KeyPoint> ToKeyPoints(const std::vector<cv::Point2f> &points) {
  std::vector<cv::KeyPoint> result;
  cv::KeyPoint::convert(points, result);
  return result;
}

std::vector<cv::KeyPoint> CombineKeypoints(
    const std::vector<cv::KeyPoint> &points_first,
    const std::vector<cv::KeyPoint> &points_second) {
  log() << "Fusion::preferFirst() call \n";

  std::vector<cv::KeyPoint> fused_keypoints;
  std::set_union(points_first.begin(), points_first.end(),
                 points_second.begin(), points_second.end(),
                 std::back_inserter(fused_keypoints),
                 [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                   return a.class_id < b.class_id;
                 });

  log() << "Fusion::preferFirst() return \n";
  return fused_keypoints;
}

}  // namespace jg
