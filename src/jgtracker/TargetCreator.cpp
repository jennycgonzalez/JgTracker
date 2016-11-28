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

#include "jgtracker/TargetCreator.h"

#include <fstream>
#include "jgtracker/operations.h"
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"

namespace jg {

namespace {

double id = 0;

cv::RNG random_number_generator_ = cv::RNG(0xFFFFFFFF);

cv::Scalar RandomColor(cv::RNG &rng) {
  unsigned color = static_cast<unsigned>(rng);
  return cv::Scalar(color & 255, (color >> 8) & 255, (color >> 16) & 255);
}

const cv::Scalar  kgreen(0,150,0);

}  // namespace

void AddTargetFromFile(
    std::vector<Target> &targets, const boost::filesystem::path &filename,
    const Mat3Uchar &frame) {
  MT_ASSERT_EQ(id, 0);
  MT_ASSERT_TRUE(targets.empty());
  MT_ASSERT_FALSE(frame.empty());

  std::ifstream stream(filename.string());
  mt::check(stream.is_open(), "Could not open '%s'", filename.c_str());

  std::string line;
  std::getline(stream, line);
  mt::check(static_cast<bool>(stream), "Fetching line from '%s' failed", filename.c_str());

  // Check what happens if file has incomplete coordinates
  std::istringstream iss(line);
  cv::Rect2d box;
  iss >> box.x >> box.y >> box.width >> box.height;
  MT_ASSERT_GE(box.x, 0);
  MT_ASSERT_GE(box.y, 0);
  MT_ASSERT_GT(box.width, 0);
  MT_ASSERT_GT(box.height, 0);
  MT_ASSERT_LT(box.br().x, frame.cols);
  MT_ASSERT_LT(box.br().y, frame.rows);
  auto color = kgreen;
  targets.emplace_back(id, color, box);
  log() << "Target box: " << targets.back().bounding_box << "\n";
}

void AddNewTarget(std::vector<Target> &targets,
                                              const cv::Rect2f &box,
                                              const Mat3Uchar &frame) {
  MT_ASSERT_FALSE(frame.empty());
  if ((box.x > 0) && (box.y > 0) && (box.width > 0) && (box.height > 0) &&
      (box.br().x < frame.cols) && (box.br().y < frame.rows)) {
    auto color = RandomColor(random_number_generator_);
    targets.emplace_back(id, color, box);
  }
}

}  // namespace jg
