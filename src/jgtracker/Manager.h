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

#ifndef JG_MANAGER_H
#define JG_MANAGER_H

#include <set>
#include <opencv2/core.hpp>
#include "jgtracker/types.h"
#include "jgtracker/Target.h"
#include "jgtracker/TargetCreator.h"
#include "jgtracker/TargetSelector.h"

namespace jg {

//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

class Display {
 public:
  Display(const std::string& config_file);
  virtual ~Display() = default;

  virtual void ShowAllTargets(const std::vector<Target>& targets,
                              Mat3Uchar& frame,
                              const std::string& window_name) const = 0;
};

class InactiveDisplay : public Display {
 public:
  InactiveDisplay(const std::string& config_file) : Display(config_file) {}


  void ShowAllTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                      const std::string& window_name) const override;
};

class ActiveDisplay : public Display {
 public:
  ActiveDisplay(const std::string& config_file) : Display(config_file) {}

  void ShowAllTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                      const std::string& window_name) const override;
};

//------------------------------------------------------------------------------
// Manager
//------------------------------------------------------------------------------

class Manager {
 public:
  Manager() = default;
  Manager(Manager&&) = default;
  Manager& operator=(Manager&&) = default;
  Manager(int delay,           const std::string& config_file);

  const Display* display() const { return display_.get(); }

 private:
  std::unique_ptr<Display> display_;
};

}  // namespace jg

#endif  // JG_MANAGER_H
