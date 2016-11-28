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
