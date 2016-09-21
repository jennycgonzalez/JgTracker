#include "jg/Manager.h"

#include <boost/property_tree/ini_parser.hpp>
#include <opencv2/highgui.hpp>
#include "jg/operations.h"

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

  min_visible_count_ = config.get<std::size_t>("Target.min_visible_count");
  max_continuous_invisible_count_ =
      config.get<std::size_t>("Target.max_continuous_invisible_count");
}

void InactiveDisplay::ShowPotentialTargets(
    const std::vector<Target> & /*targets*/, Mat3Uchar & /*frame*/,
    const std::string & /*window_name*/, const cv::Scalar & /*color*/) const {}

void InactiveDisplay::ShowTargets(const std::vector<Target> & /*targets*/,
                                  Mat3Uchar & /*frame*/,
                                  const std::string & /*window_name*/) const {}

void InactiveDisplay::ShowAllTargets(
    const std::vector<Target> & /*targets*/, Mat3Uchar & /*frame*/,
    const std::string & /*window_name*/) const {}

void ActiveDisplay::ShowPotentialTargets(const std::vector<Target> &detections,
                                         Mat3Uchar &frame,
                                         const std::string &window_name,
                                         const cv::Scalar &color) const {
  for (const auto &detection : detections) {
    detection.PrintInImage(frame, color);
  }
  auto im_size = frame.size();
  ShowImage(frame, window_name, kUpLeft.x, kUpLeft.y);
}

void ActiveDisplay::ShowTargets(const std::vector<Target> &targets,
                                Mat3Uchar &frame,
                                const std::string &window_name) const {
  for (const auto &target : targets) {
    if (target.consecutive_invisible_count < max_continuous_invisible_count_ &&
        target.total_visible_counts > min_visible_count_) {
      target.PrintInImage(frame);
    }
  }
  auto im_size = frame.size();
  ShowImage(frame, window_name, kUpLeft.x + im_size.width,
            kUpLeft.y + im_size.height);
}

void ActiveDisplay::ShowAllTargets(const std::vector<Target> &targets,
                                   Mat3Uchar &frame,
                                   const std::string &window_name) const {
  for (const auto &target : targets) {
    target.PrintInImage(frame);
  }
  auto im_size = frame.size();
  ShowImage(frame, window_name, kUpLeft.x, kUpLeft.y);
}

//------------------------------------------------------------------------------
// Manager
//------------------------------------------------------------------------------

Manager::Manager(int delay, const std::string &config_file) {
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);

  max_continuous_invisible_count_ =
      config.get<std::size_t>("Target.max_continuous_invisible_count");
  age_threshold_ = config.get<std::size_t>("Target.age_threshold");
  max_invisibility_ratio_ =
      config.get<double>("Target.max_invisibility_ratio");

  if (delay == kInactive) {
    display_.reset(new InactiveDisplay(config_file));
  } else {
    display_.reset(new ActiveDisplay(config_file));
  }

}

void Manager::DeleteLostTargets(std::vector<Target> &targets) const {
  if (!targets.empty()) {
    std::sort(
        targets.begin(), targets.end(), [](const Target &a, const Target &b) {
          return a.consecutive_invisible_count < b.consecutive_invisible_count;
        });

    //     Delete tracks that were very often invisible
    while (!targets.empty() &&
           targets.back().consecutive_invisible_count >=
               max_continuous_invisible_count_) {
      targets.pop_back();
    }

    // Delete tracks that are too often invisible
    const auto is_often_invisible = [&](const Target &target) {
      auto invisibility_ratio =
          static_cast<double>(target.consecutive_invisible_count) /
          static_cast<double>(target.age);

      return (target.age >= age_threshold_) &&
             (invisibility_ratio > max_invisibility_ratio_);
    };

    targets.erase(
        std::remove_if(targets.begin(), targets.end(), is_often_invisible),
        targets.end());
  }
}

}  // namespace jg
