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

#ifndef JG_TARGET_CREATOR_H
#define JG_TARGET_CREATOR_H

#include "jgtracker/types.h"
#include "jgtracker/Target.h"

namespace jg {

void AddNewTarget(std::vector<Target>& targets, const cv::Rect2f& box,
                  const Mat3Uchar& frame);

void AddTargetFromFile(std::vector<Target>& targets,
                       const boost::filesystem::path& filename,
                       const Mat3Uchar& frame);

}  // namespace jg

#endif  // JG_TARGET_CREATOR_H
