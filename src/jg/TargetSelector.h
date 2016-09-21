#ifndef JG_TARGET_SELECTOR_H
#define JG_TARGET_SELECTOR_H

#include <vector>
#include <opencv2/core.hpp>
#include "jg/types.h"
#include "jg/Target.h"

namespace jg {

class TargetSelector {
 public:
  TargetSelector(const Mat3Uchar& frame, const std::string& window_name) {
      frame.copyTo(user_data_.frame);
      user_data_.window_name = window_name;
  }

  cv::Rect2f GetUserSelectedTargetBox();

 private:
  UserData user_data_;
};

}  // namespace jg

#endif  // JG_TARGET_SELECTOR_H
