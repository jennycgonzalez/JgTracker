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


#include "jgtracker/TargetSelector.h"

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace jg {

namespace {

const cv::Scalar kWindowSelectionColor(100, 255, 0);
const cv::Point kUpLeft(50, 50);  // Corner where image windows start to
                                  // be displayed

void ScreenLog(cv::Mat im_draw, const std::string text) {
  int font = cv::FONT_HERSHEY_SIMPLEX;
  float font_scale = 0.5;
  int thickness = 1;
  int baseline;

  cv::Size text_size =
      cv::getTextSize(text, font, font_scale, thickness, &baseline);

  cv::Point bl_text = cv::Point(0, text_size.height);
  cv::Point bl_rect = bl_text;

  bl_rect.y += baseline;

  cv::Point tr_rect = bl_rect;
  tr_rect.x = im_draw.cols;
  tr_rect.y -= text_size.height + baseline;

  cv::rectangle(im_draw, bl_rect, tr_rect, cv::Scalar(255, 0, 0), -1);
  cv::putText(im_draw, text, bl_text, font, font_scale,
              cv::Scalar(255, 255, 255));
}

void MouseCallback(int event, int x, int y, int /* flags */, void* userdata) {
  UserData* data = reinterpret_cast<UserData*>(userdata);  
  if (data->frame.empty()) return;

  cv::Mat temp_image;
  data->frame.copyTo(temp_image);

  if (event == CV_EVENT_LBUTTONUP) {
    if (data->drawing_box == false) {
      data->origin = cv::Point2i(x, y);
      data->drawing_box = true;
    } else if (data->drawing_box == true) {
      data->drawing_box = false;
      data->box = cv::Rect2i(data->origin, cv::Point2i(x, y));
      if (data->box.width > 0 && data->box.height > 0) {
        data->ready_shape = true;
      }
    }
  }

  if (!data->drawing_box) {
    ScreenLog(temp_image, "Click on the top left corner of the object");
  } else {
    cv::rectangle(temp_image, data->origin, cv::Point(x, y),
                  cv::Scalar(255, 0, 0));

    if (!data->ready_shape){
      ScreenLog(temp_image, "Click on the bottom right corner of the object");
    }
  }

  cv::imshow(data->window_name,temp_image);
}

}  // namespace

cv::Rect2f TargetSelector::GetUserSelectedTargetBox() {
  cv::namedWindow(user_data_.window_name, CV_WINDOW_AUTOSIZE);
  cv::moveWindow(user_data_.window_name, kUpLeft.x, kUpLeft.y);
  cv::setMouseCallback(user_data_.window_name, MouseCallback, &user_data_);

  while (!user_data_.ready_shape) {
    cvWaitKey(10);
  }

  //Stop listening
  cv::setMouseCallback(user_data_.window_name, NULL);

  cv::Rect2f box = user_data_.box;
  user_data_.Reset();
  return box;
}

}  // namespace jg
