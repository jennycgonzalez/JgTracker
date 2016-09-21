#include "jg/TargetSelector.h"

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
