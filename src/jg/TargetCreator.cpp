#include "jg/TargetCreator.h"

#include <fstream>
#include <mt/assert.h>
#include <mt/check.h>
#include "jg/operations.h"

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
  mt::check(stream, "Fetching line from '%s' failed", filename.c_str());

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
