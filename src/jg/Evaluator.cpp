#include "Evaluator.h"

#include <fstream>
#include <boost/filesystem/operations.hpp>
#include <mt/assert.h>
#include <mt/check.h>
#include "jg/operations.h"
#include "jg/types.h"

namespace jg {

//------------------------------------------------------------------------------
// Single object tracking result
// http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
//------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &os, const Instances &track) {
  for (const Instance &instance : track) {
    cv::Rect box = instance.box;
    cv::Point centroid = ComputeCentroid(box);

    os << std::round(centroid.x) << ' ' << std::round(centroid.y) << ' ' <<
          std::round(box.tl().x) << ' ' << std::round(box.tl().y) << ' '
       << std::round(box.width) << ' ' << std::round(box.height) << '\n';
  }
  return os;
}

std::vector<double> ComputeCenterLocationErrorByFrame(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track) {
  MT_ASSERT_GE(gt_track.size(), track.size());
  std::vector<double> errors;
  for (size_t i = 0; i != track.size(); i++) {
    cv::Point2f track_center = ComputeCentroid(track.at(i));
    cv::Point2f gt_center = ComputeCentroid(gt_track.at(i));
    errors.push_back(ComputeEuclideanDistance(track_center, gt_center));
  }
  return errors;
}

// Refer to: Wu, Lim, Yang 2013 Online Object Tracking: A Benchmark
std::vector<double> ComputePascalChallengeScores(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track, size_t threshold_steps) {
  MT_ASSERT_GE(gt_track.size(), track.size());

  std::vector<double> frame_scores;
  for (size_t frame_num = 0; frame_num != track.size(); frame_num++) {
    cv::Rect2f intersection = track.at(frame_num) & gt_track.at(frame_num);
    float union_area = track.at(frame_num).area() +
                       gt_track.at(frame_num).area() - intersection.area();
    MT_ASSERT_GT(union_area, 0);
    frame_scores.push_back(intersection.area() / union_area);
  }

  double threshold_delta = 1.0 / static_cast<double>(threshold_steps);
  std::vector<double> scores;
  for (size_t step = 0; step != threshold_steps; step++) {
    auto score_threshold = threshold_delta * step;
    double num_successful_frames = 0;
    for (const auto& frame_score : frame_scores) {
      if (frame_score >= score_threshold) {
        ++num_successful_frames;
      }
    }
    scores.push_back(num_successful_frames / frame_scores.size());
  }
  return scores;
}

void EvaluateTrackingResults(
    const std::string& video_path, const std::string& gt_path,
    const std::string& results_path_and_suffix) {
  log() << "Start: EvaluateTrackingResults \n";

  FrameIterator iter(video_path);
  Mat3Uchar frame;
  if (iter.hasNext()) {
    iter.next().copyTo(frame);
  }
  // Read result track and gt track
  // Single target case
  auto gt_track = ReadTrackFromFile(gt_path, frame);
  std::string track_result = results_path_and_suffix + std::string(".txt");
  mt::check(boost::filesystem::is_regular_file(track_result),
            "The file %s does not exist", track_result.c_str());
  log() << "a: EvaluateTrackingResults \n";
  auto track = ReadTrackFromFile(track_result, frame);

  log() << "b: EvaluateTrackingResults \n";
  size_t threshold_steps = 100;
  auto scores = ComputePascalChallengeScores(track, gt_track, threshold_steps);
  auto pascal_scores_path =
      results_path_and_suffix + std::string("-pascal_scores.txt");
  log() << "c: EvaluateTrackingResults \n";
  SavePascalChallengeScoresInFile(scores, pascal_scores_path);

  auto errors = ComputeCenterLocationErrorByFrame(track, gt_track);
  auto center_error_path =
      results_path_and_suffix + std::string("-center-location-errors.txt");
  log() << "d: EvaluateTrackingResults \n";
  SaveCenterLocationErrorInFile(errors, center_error_path);
  log() << "End: EvaluateTrackingResults \n";
}

std::vector<cv::Rect2f> ReadTrackFromFile(
    const boost::filesystem::path& filename, const Mat3Uchar& frame) {
  std::ifstream stream(filename.string());
  mt::check(stream.is_open(), "Could not open '%s'", filename.c_str());
  std::vector<cv::Rect2f> track;

  for (std::string line; std::getline(stream, line);) {
    mt::check(stream, "Fetching line from '%s' failed", filename.c_str());
    std::istringstream iss(line);
    cv::Rect2f box;
    iss >> box.x >> box.y >> box.width >> box.height;
    MT_ASSERT_GE(box.x, 0);
    MT_ASSERT_GE(box.y, 0);
    MT_ASSERT_GE(box.width, 0);
    MT_ASSERT_GE(box.height, 0);
    MT_ASSERT_LE(box.x + box.width, frame.cols);
    MT_ASSERT_LE(box.y + box.height, frame.rows);
    track.push_back(box);
  }
  return track;
}

void SavePascalChallengeScoresInFile(
    const std::vector<double>& scores,
    const boost::filesystem::path& output_path) {
  std::ofstream ofs(output_path.c_str());
  size_t num_steps = scores.size();
  mt::check(ofs, "Error by creating file '%s'", output_path.c_str());
  ofs << "Threshold step is: " << 1.0 / static_cast<double>(num_steps) << '\n';
  mt::check(ofs << scores, "Error writing '%s'", output_path.c_str());
}

void SaveCenterLocationErrorInFile(
    const std::vector<double>& errors,
    const boost::filesystem::path& output_path) {
  std::ofstream ofs(output_path.c_str());
  mt::check(ofs, "Error by creating file '%s'", output_path.c_str());
  mt::check(ofs << errors, "Error writing '%s'", output_path.c_str());
}

void SaveExecutionTime(double execution_time,
                                   const std::string &output_path) {
  std::ofstream ofs(output_path.c_str());
  mt::check(ofs, "Error by creating file '%s'", output_path.c_str());
  mt::check(ofs << execution_time, "Error writing '%s'", output_path.c_str());
}

void SaveTracksInFile(const Instances &track,
                                  const std::string &output_path) {
  log() << "Start: Saving tracks in File\n";
  MT_ASSERT_FALSE(track.empty());
  log() << "Saving results\n";
  std::ofstream ofs(output_path.c_str());
  mt::check(ofs, "Error by creating file '%s'", output_path.c_str());
  mt::check(ofs << track, "Error writing '%s'", output_path.c_str());
  log() << "End of Saving results\n";
}

void UpdateTracks(const std::vector<Target> &targets,
                              const cv::Rect2f &frame_window,
                              Instances &track) {
  MT_ASSERT_LE(targets.size(), 1);
  if (!targets.empty()) {
    // Truncate bounding box to keep the area inside the image frame
    cv::Rect2f target_box = targets.back().bounding_box;
    cv::Rect2f corrected_box = frame_window & target_box;
    MT_ASSERT_GE(corrected_box.x, 0);
    MT_ASSERT_GE(corrected_box.y, 0);
    MT_ASSERT_LE(corrected_box.br().x, frame_window.width);
    MT_ASSERT_LE(corrected_box.br().y, frame_window.height);

    track.emplace_back(targets.back().id, corrected_box);
  }
}

}  // namespace jg
