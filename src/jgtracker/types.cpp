#include "jgtracker/types.h"

#include <algorithm>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/operations.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"

namespace jg {

namespace fs = boost::filesystem;

namespace {

const auto kImageExtensions = {".bmp", ".BMP", ".dib",  ".DIB",  ".gif", ".GIF",
                               ".jpg", ".JPG", ".jpeg", ".JPEG", ".pbm", ".PBM",
                               ".pgm", ".PGM", ".png",  ".PNG",  ".ppm", ".PPM",
                               ".tif", ".TIF", ".tiff", ".TIFF"};

bool IsImageFile(const fs::path &path) {
  if (fs::is_regular_file(path)) {
    return std::find(std::begin(kImageExtensions), std::end(kImageExtensions),
                     path.extension()) != std::end(kImageExtensions);
  }
  return false;
}

std::vector<fs::path> GetImageFiles(const fs::path &directory) {
  mt::check(fs::is_directory(directory), "'%s' is not a folder",
            directory.c_str());
  std::vector<fs::path> paths;
  fs::directory_iterator end;
  for (fs::directory_iterator it(directory); it != end; ++it) {
    if (IsImageFile(it->path())) {
      paths.push_back(fs::canonical(it->path()));
      // bfs::canonical returns an absolute path beginning with '/'.
      // bfs::absolute also returns an absolute path, but which may
      // contain relative elements such as "..".
    }
  }

  // Assert that all paths are of the same file type.
  const auto has_same_extension = [&paths](const fs::path &path) {
    return path.extension() == paths.front().extension();
  };
  mt::check(std::all_of(paths.begin(), paths.end(), has_same_extension),
            "Not all images in folder '%s' have the same file extension",
            directory.c_str());
  return paths;
}

// TODO Check if no object reuse is a bottleneck
std::vector<cv::Point2f> ToPoints(const std::vector<cv::KeyPoint>& keypoints) {
  std::vector<cv::Point2f> result;
  cv::KeyPoint::convert(keypoints, result);
  return result;
}

}  // namespace

std::ostream &operator<<(std::ostream &os, const VecDouble &vector) {
  for (double value : vector) {
    os << value << '\n';
  }
  return os;
}

//------------------------------------------------------------------------------
// FrameIterator
//------------------------------------------------------------------------------

FrameIterator::FrameIterator(const fs::path &directory)
    : paths_(GetImageFiles(directory)) {
  // Sanity check
  mt::check(std::all_of(paths_.begin(), paths_.end(), IsImageFile),
            "Not all files are images");
  std::sort(paths_.begin(), paths_.end(), std::greater<fs::path>());
}

bool FrameIterator::hasNext() const { return !paths_.empty(); }

Mat3Uchar FrameIterator::next() {
  mt::check(hasNext(), "There are not more frames to retrieve");

  const auto next_path = paths_.back();
  paths_.pop_back();

  Mat3Uchar next_frame = cv::imread(next_path.string(), CV_LOAD_IMAGE_COLOR);
  mt::check(next_frame.data, "Could not open or find the image");

  return next_frame;
}

std::vector<cv::KeyPoint> Tracker::TrackKeypointsWithOpticalFlow(
    const cv::Mat previous_frame, const cv::Mat current_frame,
    const std::vector<cv::KeyPoint> &keypoints) {
  std::vector<cv::KeyPoint> tracked_keypoints;

  if (!keypoints.empty()) {
    std::vector<unsigned char> found;
    std::vector<float> err;  // Needs to be float

    std::vector<cv::Point2f> points = ToPoints(keypoints);
    std::vector<cv::Point2f> forward_tracked_points;
    // Calculate forward optical flow for prev_location
    cv::calcOpticalFlowPyrLK(previous_frame, current_frame, points,
                             forward_tracked_points, found, err);

    std::vector<cv::Point2f> backward_tracked_points;
    std::vector<unsigned char> found_back;
    std::vector<float> err_back;  // Needs to be float

    // Calculate backward optical flow for prev_location
    cv::calcOpticalFlowPyrLK(current_frame, previous_frame,
                             forward_tracked_points, backward_tracked_points,
                             found_back, err_back);

    MT_ASSERT_EQ(points.size(), backward_tracked_points.size());

    for (size_t k = 0; k != backward_tracked_points.size(); k++) {
      if (found_back.at(k) == 1) {
        cv::Point2f a = backward_tracked_points.at(k);
        cv::Point2f b = points.at(k);
        float l2norm = cv::norm(a - b);
        if (l2norm < threshold_found_backwards_) {
          float size = keypoints.at(k).size;
          tracked_keypoints.emplace_back(forward_tracked_points.at(k), size);
          tracked_keypoints.back().class_id = keypoints.at(k).class_id;
        }
      }
    }
  }
  return tracked_keypoints;
}

}  // namespace jg
