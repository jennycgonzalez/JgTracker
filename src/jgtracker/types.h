#ifndef JG_TYPES_H
#define JG_TYPES_H

#include <vector>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

namespace jg {

typedef cv::Mat_<uchar> MatUchar;
typedef cv::Mat_<cv::Vec3b> Mat3Uchar;
typedef cv::Mat_<float> MatFloat;
typedef cv::Mat_<cv::Vec3f> Mat3Float;
typedef cv::Mat_<double> MatDouble;
typedef std::vector<double> VecDouble;

std::ostream& operator<<(std::ostream& os, const VecDouble& vector);

enum InitializerEnum { kManual, kFile };

enum KeypointDetectorEnum {
  kBRISKDetector = 0,
  kFASTDetector,
  kORBDetector,
  kSIFTDetector,
  kSURFDetector
};

enum KeypointDescriptorEnum {
  kBRISKDescriptor = 0,
  kORBDescriptor,
  kSIFTDescriptor,
  kSURFDescriptor
};

enum HistogramEnum {
  kIntegralBMHistogram = 0,
  kNormalHistogram
};

enum ColorSpaceEnum {
  kHSV = 0,
  kGrayscale
};

//------------------------------------------------------------------------------
// SearchArea
//------------------------------------------------------------------------------

struct Background {
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keypoints;
};

//------------------------------------------------------------------------------
// SearchArea
//------------------------------------------------------------------------------

struct SearchArea {
  cv::RotatedRect roi;
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Point2f> points;
};
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

//------------------------------------------------------------------------------
// FrameIterator
//------------------------------------------------------------------------------

class FrameIterator {
  // Class to iterate over a sequence of `Frame`s which are instantiated
  // on demand for memory-friendliness.

 public:
  FrameIterator() = default;

  FrameIterator(const boost::filesystem::path& directory);

  bool hasNext() const;

  Mat3Uchar next();

 private:
  std::vector<boost::filesystem::path> paths_;
};

//------------------------------------------------------------------------------
// User interaction
//------------------------------------------------------------------------------

struct UserData {
  Mat3Uchar frame;
  std::string window_name;
  // Selection flag: it is active while the mouse button is pressed, and
  // dragged
  // inside the target window
  bool drawing_box = false;
  // Ready flag: it is 0 when the shape has been selected with the mouse
  bool ready_shape = false;
  // Upper-left corner of the rectangle: this is set at the beginning of the
  // selection
  cv::Point2i origin = cv::Point2d(-1, -1);
  // Output: the selected region
  cv::Rect2i box = cv::Rect2i(-1, -1, 0, 0);

  void Reset() {
    ready_shape = false;
    origin = cv::Point2d(-1, -1);
    box = cv::Rect2i(-1, -1, 0, 0);
  }
};

//------------------------------------------------------------------------------
// Tracker
//------------------------------------------------------------------------------

class Tracker {
 public:
  Tracker() : threshold_found_backwards_(30) {}
  std::vector<cv::KeyPoint> TrackKeypointsWithOpticalFlow(
      const cv::Mat im_prev, const cv::Mat im_gray,
      const std::vector<cv::KeyPoint>& keypoints);

 private:
  float threshold_found_backwards_;
};

}  // namespace jg

#endif  // JG_TYPES_H
