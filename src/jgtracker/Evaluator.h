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

#ifndef JG_EVALUATOR_H
#define JG_EVALUATOR_H

#include <vector>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/types.hpp>
#include "jgtracker/types.h"
#include "jgtracker/Target.h"

namespace jg {

struct Instance {
  Instance() = default;
  Instance(const Instance&) = default;
  Instance(Instance&&) = default;
  Instance& operator=(Instance&&) = default;

  Instance(size_t target_id, const cv::Rect2f& t_box): id(target_id), box(t_box) {}

  size_t id;
  cv::Rect2f box;
};

struct Instances : std::vector<Instance> {};

std::ostream& operator<<(std::ostream& os, const Instances& track);

std::vector<double> ComputePascalChallengeScores(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track, size_t threshold_steps);

std::vector<double> ComputeCenterLocationErrorByFrame(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track);

void EvaluateTrackingResults(const std::string& video_path,
                             const std::string& gt_path,
                             const std::string& results_path_and_suffix);

std::vector<cv::Rect2f> ReadTrackFromFile(
    const boost::filesystem::path& filename, const Mat3Uchar& frame);

void SavePascalChallengeScoresInFile(
    const std::vector<double>& scores,
    const boost::filesystem::path& output_path);
void SaveCenterLocationErrorInFile(const std::vector<double>& errors,
                                   const boost::filesystem::path& output_path);

void SaveExecutionTime(double execution_time, const std::string& output_path);

void SaveTracksInFile(const Instances& track, const std::string& output_path);

void UpdateTracks(const std::vector<Target>& targets,
                  const cv::Rect2f& frame_window, Instances& track);

}  // namespace jg

#endif  // JG_EVALUATOR_H
