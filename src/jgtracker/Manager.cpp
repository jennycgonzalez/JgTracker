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

#include "jgtracker/Manager.h"

#include <boost/property_tree/ini_parser.hpp>
#include <opencv2/highgui.hpp>
#include "jgtracker/operations.h"

namespace jg {

namespace {

const int kInactive = -1;
const cv::Point kUpLeft(2000,200);  // Corner where image windows start to
                                  // be displayed
}  // namespace

//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

Display::Display(const std::string &config_file) {
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);
}

void InactiveDisplay::ShowAllTargets(
    const std::vector<Target> & /*targets*/, Mat3Uchar & /*frame*/,
    const std::string & /*window_name*/) const {}

void ActiveDisplay::ShowAllTargets(const std::vector<Target> &targets,
                                   Mat3Uchar &frame,
                                   const std::string &window_name) const {
  for (const auto &target : targets) {
    target.PrintInImage(frame);
  }
  ShowImage(frame, window_name, kUpLeft.x, kUpLeft.y);
}

//------------------------------------------------------------------------------
// Manager
//------------------------------------------------------------------------------

Manager::Manager(int delay, const std::string &config_file) {
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);

  if (delay == kInactive) {
    display_.reset(new InactiveDisplay(config_file));
  } else {
    display_.reset(new ActiveDisplay(config_file));
  }

}

}  // namespace jg
