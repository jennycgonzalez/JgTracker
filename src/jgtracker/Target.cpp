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
Copyright (c) 2016, Jenny Gonzalez
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

#include "jgtracker/Target.h"

#include "jgtracker/thirdparty/mt/assert.h"
#include <opencv2/imgproc.hpp>
#include "jgtracker/operations.h"

namespace jg {

namespace {

const int kLineThickness = 2;

}  // namespace

Target::Target(size_t id, const cv::Scalar &color, const cv::Rect2f &box)
    : id(id), label_color(color) {
  log() << "Started creating target \n";
  MT_ASSERT_GT(box.area(), 0);
  bounding_box = box;
  initial_size = bounding_box.size();
  log() << "Finish creating target \n";
}

void Target::ChangeBoundingBox(const cv::Rect2f &new_bounding_box) {
  log() << "Started changing target bounding box \n";
  bounding_box = new_bounding_box;
  log() << "Finished changing target bounding box \n";
}

void Target::Translate(const cv::Point2f &new_center,
                       const cv::Rect2f &/*frame_window*/) {
  log() << "Start relocating target with centroid \n";
  cv::Point2f current_centroid = ComputeCentroid(bounding_box);
  cv::Point2f delta = new_center - current_centroid;
  bounding_box = bounding_box + delta;
//  cv::Rect2f temp_box = bounding_box & frame_window;
//  bounding_box = temp_box;
  log() << "Finish relocating target with centroid \n";
}

void Target::Resize(float width, float height) {
  log() << "Start resizing target with width and height \n";

  MT_ASSERT_GT(height, 0);
  MT_ASSERT_GT(width, 0);

  cv::Rect2f temp_box = cv::Rect2f(bounding_box.tl(), cv::Size(width, height));
  bounding_box = temp_box;
  log() << "Finish resizing target with width and height \n";
}

std::vector<cv::Point2f> Target::ComputeCorners() const {
  std::vector<cv::Point2f> corners;
  corners.push_back(bounding_box.tl());
  corners.emplace_back(bounding_box.tl().x, bounding_box.br().y);
  corners.push_back(bounding_box.br());
  corners.emplace_back(bounding_box.br().x, bounding_box.tl().y);
  return corners;
}

const cv::Size2i Target::GetSize() const {
  log() << "Start get target size \n";
  return cv::Size2i(bounding_box.size());
}

void Target::UpdateCurrentHistogram(const Mat3Uchar &image) {
  cv::Mat roi_mask(image.size(), CV_8UC1, cv::Scalar::all(0));
  roi_mask(bounding_box).setTo(cv::Scalar::all(255));
  current_histogram_p->Update(image, roi_mask);
}

void Target::InitHistograms(ColorSpaceEnum color_space_enum,
                            const cv::Mat &image) {
  switch (color_space_enum) {
    case kGrayscale:
      original_histogram_p.reset(new GrayscaleHistogram);
      current_histogram_p.reset(new GrayscaleHistogram);
      break;
    case kHSV:
      original_histogram_p.reset(new HsvHistogram);
      current_histogram_p.reset(new HsvHistogram);
      break;
    default:
      break;
  }

  cv::Mat roi_mask(image.size(), CV_8UC1, cv::Scalar::all(0));
  roi_mask(bounding_box).setTo(cv::Scalar::all(255));
  original_histogram_p->Update(image, roi_mask);
  current_histogram_p->Update(image, roi_mask);
}

void Target::SaveOriginalImage(const Mat3Uchar &image) {
  image(bounding_box).copyTo(original_image);
  MT_ASSERT_LT(original_image.size().height, image.size().height);
  MT_ASSERT_LT(original_image.size().width, image.size().width);
}

void Target::PrintInImage(Mat3Uchar &image) const {
  log() << "Start: Print Target in image \n";
  cv::Rect2f image_frame(0, 0, image.cols, image.rows);
  cv::Rect2f box = bounding_box & image_frame;
  if (box.area() > 0) {
    cv::rectangle(image, cv::Rect2i(box), label_color, kLineThickness);
    cv::putText(image, std::to_string(id), box.br(), cv::FONT_HERSHEY_SIMPLEX,
                0.5, label_color);
    log() << "End: Print Target in image \n";
  }
}

void Target::PrintInImage(Mat3Uchar &image, const cv::Scalar &color) const {
  //  auto box_2i = cv::Rect2i(bounding_box);
  cv::Rect2f image_frame(0, 0, image.cols, image.rows);
  cv::Rect2f box = bounding_box & image_frame;
  if (box.area() > 0) {
    cv::rectangle(image, cv::Rect2i(bounding_box), color, kLineThickness);
    cv::putText(image, std::to_string(id), bounding_box.br(),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
  }
}

}  // namespace jg
