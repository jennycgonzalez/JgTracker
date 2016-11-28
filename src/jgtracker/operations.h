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

#ifndef JG_OPERATIONS_H
#define JG_OPERATIONS_H

#include <fstream>
#include "jgtracker/types.h"

//#define JG_LOGGING_ENABLED

namespace jg {

void AssignUniqueIdsToKeypoints(std::vector<cv::KeyPoint>& keypoints);

float GetMedian(std::vector<float>& values);

std::vector<cv::Rect2f> CreateImageStripes(float height, float width, size_t num_patches);

void ComputeGrayscaleHistogram(const Mat3Uchar& image, const MatUchar& roi_mask,
                               cv::Mat& hist);

cv::Point2f ComputeCentroid(const cv::Rect2f &box);

void ComputeHsvHistogram(const Mat3Uchar& image, const cv::Mat& roi_mask,
                         cv::Mat& hist);


void ComputeCorners(const cv::RotatedRect& box,
                    std::vector<cv::Point2f>& corners);

double ComputeEuclideanDistance(cv::Point2d a, cv::Point2d b);

std::vector<cv::KeyPoint> ConditionKeypoints(
    const cv::Point2f& centroid, const std::vector<cv::KeyPoint>& keypoints);

void ConditionPoints(const cv::Point2f& centroid,
                     const std::vector<cv::KeyPoint>& keypoints,
                     std::vector<cv::Point2f>& conditioned_points);

void DrawKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                   const Mat3Uchar& image, const cv::Scalar& color,
                   const std::string& window_name);

void EliminateDuplicatedIDKeypoints(std::vector<cv::KeyPoint>& keypoints);

cv::Mat ReadTransitionMatrixFromFile(const boost::filesystem::path& filename);

cv::Point2f Rotate(const cv::Point2f& v, float angle);

std::vector<cv::KeyPoint> RotateAndScaleKeypoints(
    const std::vector<cv::KeyPoint>& keypoints, float rotation, float scale);

void ShowImage(const cv::Mat& frame, const std::string& name, int corner_x,
               int corner_y);

void SortKeypointsByID(std::vector<cv::KeyPoint>& keypoints);

MatUchar ToGrayscaleFrame(const Mat3Uchar& frame);

std::vector<cv::KeyPoint> ToKeyPoints(const std::vector<cv::Point2f>& points);

cv::Mat ReadTransitionMatrixFromFile(const boost::filesystem::path& filename);

cv::Mat ReadControlMatFromFile(const boost::filesystem::path& filename,
                               size_t num_rows);

std::vector<cv::KeyPoint> CombineKeypoints(
    const std::vector<cv::KeyPoint>& points_first,
    const std::vector<cv::KeyPoint>& points_second);

inline std::ostream& log() {
#ifdef JG_LOGGING_ENABLED
  return std::clog;
#else
  static std::ofstream null("/dev/null");
  return null;
#endif
}

}  // namespace jg

#endif  // JG_OPERATIONS_H
