#ifndef JG_EVALUATOR_H
#define JG_EVALUATOR_H

#include <vector>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/types.hpp>
#include "jg/types.h"
#include "jg/Target.h"

namespace jg {

struct Instance {
  Instance() = default;
  Instance(const Instance&) = default;
  Instance(Instance&&) = default;
  Instance& operator=(Instance&&) = default;

  Instance(size_t target_id, const cv::Rect2f& t_box): id(target_id), box(t_box) {}

  size_t id;
  cv::Rect2f box;
};

struct Instances : std::vector<Instance> {};

std::ostream& operator<<(std::ostream& os, const Instances& track);

std::vector<double> ComputePascalChallengeScores(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track, size_t threshold_steps);

std::vector<double> ComputeCenterLocationErrorByFrame(
    const std::vector<cv::Rect2f>& track,
    const std::vector<cv::Rect2f>& gt_track);

void EvaluateTrackingResults(const std::string& video_path,
                             const std::string& gt_path,
                             const std::string& results_path_and_suffix);

std::vector<cv::Rect2f> ReadTrackFromFile(
    const boost::filesystem::path& filename, const Mat3Uchar& frame);

void SavePascalChallengeScoresInFile(
    const std::vector<double>& scores,
    const boost::filesystem::path& output_path);
void SaveCenterLocationErrorInFile(const std::vector<double>& errors,
                                   const boost::filesystem::path& output_path);

void SaveExecutionTime(double execution_time, const std::string& output_path);

void SaveTracksInFile(const Instances& track, const std::string& output_path);

void UpdateTracks(const std::vector<Target>& targets,
                  const cv::Rect2f& frame_window, Instances& track);

}  // namespace jg

#endif  // JG_EVALUATOR_H
