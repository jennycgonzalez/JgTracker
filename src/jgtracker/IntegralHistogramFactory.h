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

#ifndef JG_INTEGRAL_HISTOGRAM_FACTORY_H
#define JG_INTEGRAL_HISTOGRAM_FACTORY_H

#include "jgtracker/types.h"

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
