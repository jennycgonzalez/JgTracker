#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <mt/check.h>
#include <mt/assert.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "jg/Evaluator.h"
#include "jg/FeaturesExtractor.h"
#include "jg/IntegralHistogramFactory.h"
#include "jg/Manager.h"
#include "jg/MatchFinder.h"
#include "jg/operations.h"
#include "jg/SIRParticleFilter.h"
#include "jg/Target.h"
#include "jg/TargetCreator.h"
#include "jg/TargetSelector.h"
#include "jg/Voting.h"

//#include <google/heap-profiler.h>
//#include <google/profiler.h>

namespace fs = boost::filesystem;

const std::string kConfigFile = "config.ini";
const std::string kGroundtruthFile = "groundtruth.txt";
const std::string kResultsRelativePath = "results/";
const std::string kFramesRelativePath = "img/";
const std::string kMethodName = "jg-result-test";

const bool kShowImages = true;

const char *TOOLNAME = "jgtracker";
const char *HELP = "help";
const char *TRACK = "track";

const std::vector<std::string> COMMANDS = {HELP, TRACK};

struct CommandLine {
  struct Error : public std::runtime_error {
    Error(const std::string &what) : std::runtime_error(what) {}
  };
  std::string command, base_path, gt_filepath, delay;
  std::string results_filename, ini_config_file;
  std::string video_path, results_path;
};

bool IsInSet(const std::string &argument, const std::vector<std::string> &set) {
  return std::find(set.begin(), set.end(), argument) != set.end();
}

void ShowHelpMessage() {
  std::printf(
      "OVERVIEW: A configurable object tracking system.\n"
      "USAGE:\n"
      "\n %s COMMAND [PATH] [PATH] [SETTING] [SETTING] [SETTING]"
      "\n\nCOMMANDS\n"
      "\n %-10s Print this help message and exit."
      "\n %-10s Run the object tracker."
      "\n\nEXAMPLES\n"
      "\n %s %-5s root/path delay\n\n",
      TOOLNAME, HELP, TRACK, TOOLNAME, TRACK);
}

CommandLine ReadCommandLine(int argc, const char **argv) {
  CommandLine cmd_line;
  const auto end = std::next(argv, argc);
  auto it = std::next(argv);

  using E = CommandLine::Error;

  mt::check<E>(it != end, "No COMMAND was given");
  mt::check<E>(IsInSet(*it, COMMANDS), "Expected COMMAND when reading '%s'",
               *it);
  cmd_line.command = *it++;

  if (cmd_line.command == TRACK) {
    mt::check<E>(it != end, "The base path was not given");
    mt::check<E>(fs::is_directory(*it), "'%s' is not a folder", *it);
    cmd_line.base_path = *it++;

    cmd_line.video_path = cmd_line.base_path;
    cmd_line.video_path += kFramesRelativePath;
    mt::check<E>(fs::is_directory(cmd_line.video_path), "'%s' is not a folder",
                 cmd_line.video_path.c_str());

    cmd_line.gt_filepath = cmd_line.base_path + kGroundtruthFile;
    mt::check<E>(fs::is_regular_file(cmd_line.gt_filepath),
                 "'%s' is not a regular file", cmd_line.gt_filepath.c_str());

    cmd_line.results_path = cmd_line.base_path + kResultsRelativePath;
    if (!fs::is_directory(cmd_line.results_path)) {
      mt::check<E>(fs::create_directory(cmd_line.results_path),
                   "Cannot create folder '%s'", cmd_line.results_path.c_str());
    }

    cmd_line.ini_config_file = cmd_line.base_path + kConfigFile;
    mt::check<E>(fs::is_regular_file(cmd_line.ini_config_file),
                 "'%s' is not a regular file",
                 cmd_line.ini_config_file.c_str());

    mt::check<E>(it != end, "The delay was not given");
    cmd_line.delay = *it;
  }
  return cmd_line;
}

jg::ColorSpaceEnum GetColorSpaceEnum(const std::string &color_space) {
  if (color_space == "hsv") {
    return jg::kHSV;
  } else {
    return jg::kGrayscale;
  }
}

jg::HistogramEnum GetHistogramEnum(const std::string &histogram_type) {
  if (histogram_type == "integral") {
    return jg::kIntegralHistogram;
  } else {
    return jg::kNormalHistogram;
  }
}

jg::InitializerEnum GetInitializationEnum(
    const std::string &initialization_mode) {
  if (initialization_mode == "file") {
    return jg::kFile;
  } else {
    return jg::kManual;
  }
}

const char *BRISK = "BRISK";
const char *FAST = "FAST";
const char *ORB = "ORB";
const char *SIFT = "SIFT";
const char *SURF = "SURF";

jg::KeypointDetectorEnum GetKeypointExtractorEnum(
    const std::string &keypoint_detector_type) {
  if (keypoint_detector_type == BRISK) {
    return jg::kBRISKDetector;
  } else if (keypoint_detector_type == FAST) {
    return jg::kFASTDetector;
  } else if (keypoint_detector_type == ORB) {
    return jg::kORBDetector;
  } else if (keypoint_detector_type == SIFT) {
    return jg::kSIFTDetector;
  } else {
    return jg::kSURFDetector;
  }
}

jg::KeypointDescriptorEnum GetKeypointDescriptorEnum(
    const std::string &keypoint_descriptor_type) {
  if (keypoint_descriptor_type == BRISK) {
    return jg::kBRISKDescriptor;
  } else if (keypoint_descriptor_type == ORB) {
    return jg::kORBDescriptor;
  } else if (keypoint_descriptor_type == SIFT) {
    return jg::kSIFTDescriptor;
  } else {
    return jg::kSURFDescriptor;
  }
}

//------------------------------------------------------------------------------
// Track with Particle Filter + CMT
//------------------------------------------------------------------------------

void TrackWithParticleFilterAndKeypoints(const CommandLine &cmd_line) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  // Set up the tracking pipeline
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(cmd_line.ini_config_file.c_str(),
                                             config);

  const std::string color_space = config.get<std::string>("Image.colorspace");
  const jg::ColorSpaceEnum color_space_enum = GetColorSpaceEnum(color_space);

  const std::string histogram_type =
      config.get<std::string>("System.histogram_type");
  const auto histogram_enum = GetHistogramEnum(histogram_type);

  const double image_scale = config.get<double_t>("Image.scale");

  const std::string initialization_mode =
      config.get<std::string>("System.initialization_mode");
  const auto initialization_enum = GetInitializationEnum(initialization_mode);

  const std::string keypoint_extractor =
      config.get<std::string>("System.keypoint_extractor");
  const auto keypoint_detector_enum =
      GetKeypointExtractorEnum(keypoint_extractor);

  const std::string keypoint_descriptor =
      config.get<std::string>("System.keypoint_descriptor");

  const size_t num_image_stripes =
      config.get<size_t>("ParticleFilter.num_stripes");

  const auto keypoint_descriptor_enum =
      GetKeypointDescriptorEnum(keypoint_descriptor);

  const int delay = std::stoi(cmd_line.delay);

  jg::Manager manager(delay, cmd_line.ini_config_file);

  jg::FeaturesExtractor feature_extractor(keypoint_detector_enum,
                                          keypoint_descriptor_enum,
                                          cmd_line.ini_config_file);
  jg::MatchFinder match_finder(keypoint_descriptor_enum,
                               cmd_line.ini_config_file);
  jg::Voting voting;

  jg::FrameIterator iter(cmd_line.video_path);

  if (iter.hasNext()) {
    jg::Mat3Uchar big_frame = iter.next();
    jg::Mat3Uchar frame;
    cv::Mat frame_gray;
    jg::Mat3Uchar previous_frame;
    cv::Mat previous_frame_gray;
    cv::resize(big_frame, frame, cv::Size(), image_scale, image_scale,
               cv::INTER_AREA);
    size_t frame_number = 1;
    cv::Rect2f frame_window(0.0, 0.0, frame.cols, frame.rows);

    std::unique_ptr<jg::SIRParticleFilters> particle_filters;
    std::unique_ptr<jg::IntegralHistogramFactory> integral_histogram_creator;

    if (histogram_enum == jg::kIntegralHistogram) {
      particle_filters.reset(new jg::SIRParticleFiltersIntegralHistogram(
          cmd_line.ini_config_file, frame));
      switch (color_space_enum) {
        case jg::kHSV:
          integral_histogram_creator.reset(new jg::HsvIntegralHistogramFactory);
          break;
        default:
          integral_histogram_creator.reset(
              new jg::GrayscaleIntegralHistogramFactory);
          break;
      }

    } else {
      particle_filters.reset(new jg::SIRParticleFiltersNormalHistogram(
          cmd_line.ini_config_file, frame));
    }
    particle_filters->SetupColorSpace(color_space_enum);

    jg::Tracker tracker;

    jg::Mat3Uchar temp_frame;
    frame.copyTo(temp_frame);

    const std::string target_window = "Target";
    if (kShowImages) {
      cv::namedWindow(target_window);
      cv::imshow(target_window, temp_frame);
    }

    std::vector<jg::Target> targets;
    jg::Instances track;
    jg::Background background_model;

    if (initialization_enum == jg::kFile) {
      jg::AddTargetFromFile(targets, cmd_line.gt_filepath, frame);
      MT_ASSERT_FALSE(targets.empty());
    } else if (initialization_enum == jg::kManual) {
      jg::TargetSelector target_selector(frame, target_window);
      cv::Rect2f selected_box = target_selector.GetUserSelectedTargetBox();
      jg::AddNewTarget(targets, selected_box, frame);
      MT_ASSERT_FALSE(targets.empty());
    }

    jg::Target &new_target = targets.back();

    new_target.SaveOriginalImage(frame);
    new_target.InitHistograms(color_space_enum, frame);

    std::vector<cv::Mat> a_channel_hist;
    std::vector<cv::Mat> b_channel_hist;
    std::vector<cv::Mat> c_channel_hist;
    if (histogram_enum == jg::kIntegralHistogram) {
      // For computing the integral histograms
      integral_histogram_creator->ComputeIntegralBinaryMasks(
          new_target.original_image, a_channel_hist, b_channel_hist,
          c_channel_hist);
      std::vector<cv::Rect2f> stripes = jg::CreateImageStripes(
          new_target.initial_size.width, new_target.initial_size.height,
          num_image_stripes);

      for (const cv::Rect2f &stripe : stripes) {
        std::vector<float> histogram_vector =
            integral_histogram_creator->CreateHistogramVector(
                stripe, a_channel_hist, b_channel_hist, c_channel_hist);
        new_target.stripes_histograms.push_back(histogram_vector);
      }
    }

    feature_extractor.UpdateFeatures(frame, new_target.bounding_box,
                                     new_target.original_keypoints,
                                     new_target.original_descriptors);
    jg::AssignUniqueIdsToKeypoints(new_target.original_keypoints);
    jg::log() << "Number of originak keypoints:"
              << new_target.original_keypoints.size() << "\n";

    // Initialize active keypoints
    new_target.active_keypoints = new_target.original_keypoints;

    cv::Point2f target_centroid = jg::ComputeCentroid(new_target.bounding_box);
    new_target.original_conditioned_keypoints =
        jg::ConditionKeypoints(target_centroid, new_target.original_keypoints);

    voting.InitializeConsensusData(new_target.original_keypoints,
                                   new_target.original_distances_pairwise,
                                   new_target.original_angles_pairwise);

    particle_filters->CreateFilter(new_target);
    particle_filters->PrintParticles(temp_frame);

    if (kShowImages) {
      manager.display()->ShowAllTargets(targets, temp_frame, "Target");
    }

    jg::log() << "Frame num:" << frame_number << "\n";

    if (delay != -1) {
      cv::waitKey(delay);
    }

    jg::log() << "Num frame channels:" << frame.channels() << "\n";
    jg::log() << "Update track:"
              << "\n";

    jg::UpdateTracks(targets, frame_window, track);
    jg::log() << "Finish  track:"
              << "\n";

    jg::log() << "Create background mask" << frame.channels() << "\n";
    cv::Mat background_mask = cv::Mat::ones(frame.rows, frame.cols, CV_8UC1);
    background_mask(new_target.bounding_box).setTo(cv::Scalar::all(0));
    feature_extractor.UpdateFeatures(frame, background_mask,
                                     background_model.keypoints,
                                     background_model.descriptors);

    jg::log() << "Finish creating background mask" << frame.channels() << "\n";

    if (iter.hasNext()) {
      frame.copyTo(previous_frame);

      cv::cvtColor(previous_frame, previous_frame_gray, cv::COLOR_BGR2GRAY);
      iter.next().copyTo(frame);
      frame.copyTo(temp_frame);
      cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
      ++frame_number;
    }

    bool can_continue = true;
    // We are handling one single target so far
    jg::Target &target = targets.back();

    do {
      const float nan = std::numeric_limits<float>::quiet_NaN();
      jg::log() << "Frame num:" << frame_number << "\n";

      // Track with Particle Filter first
      particle_filters->DriftAndDiffuse();

      //     particle_filters->PrintParticles(extra_frame_2);
      //     cv::imshow("Diffused Particles", extra_frame_2);

      // Weight particles according to importance
      std::chrono::time_point<std::chrono::system_clock> start_function,
          end_function;
      start_function = std::chrono::system_clock::now();
      particle_filters->ComputeNormalizedWeights(frame);
      end_function = std::chrono::system_clock::now();

      std::chrono::duration<double> function_time =
          end_function - start_function;
      jg::log() << "Elapsed time weighting: " << function_time.count() << "s\n";

      // Here the algorithm splits into two branches depending on the
      // confidence of the particle filter
      cv::Point2f particles_centroid(nan, nan);

      if (particle_filters->GetFilter(0).accumulated_weight > 0) {
        particles_centroid =
            particle_filters->GetFilter(0)
                .ComputeParticlesCentroid(frame.cols, frame.rows);

        particle_filters->ResampleParticles();

        // Keypoints Tracking Module
        cv::Point2f centroid(nan, nan);
        float scale = nan;
        float rotation = nan;
        float new_width = nan;
        float new_height = nan;

        std::vector<cv::KeyPoint> tracked_keypoints =
            tracker.TrackKeypointsWithOpticalFlow(
                previous_frame_gray, frame_gray, target.active_keypoints);

        //      cv::Scalar green = cv::Scalar(0, 255, 0);
        //      jg::DrawKeypoints(tracked_keypoints, frame, green,
        //                        "First tracked keypoints");

        //   Keep only tracked keypoints that are inside the bounding box
        std::vector<cv::KeyPoint> tracked_keypoints_inside;

        float tlx =
            particles_centroid.x - std::floor(target.bounding_box.width * 0.5);
        float brx =
            particles_centroid.x + std::floor(target.bounding_box.width * 0.5);
        float tly =
            particles_centroid.y - std::floor(target.bounding_box.height * 0.5);
        float bry =
            particles_centroid.y + std::floor(target.bounding_box.height * 0.5);

        cv::Rect2f particle_filter_box(cv::Point2f(tlx, tly),
                                       cv::Point2f(brx, bry));
        for (const cv::KeyPoint &keypoint : tracked_keypoints) {
          if (particle_filter_box.contains(keypoint.pt)) {
            tracked_keypoints_inside.push_back(keypoint);
          }
        }

        jg::SortKeypointsByID(tracked_keypoints_inside);
        jg::EliminateDuplicatedIDKeypoints(tracked_keypoints_inside);

        voting.EstimateScaleRotation(
            tracked_keypoints_inside, target.original_distances_pairwise,
            target.original_angles_pairwise, scale, rotation);

        if (!std::isnan(scale) && !std::isnan(rotation)) {
          // First estimated bounding box
          // with center computed with Particle Filter
          // and scale obtained with tracked keypoints
          cv::Size o_size = target.initial_size;
          new_height = o_size.height * scale;
          new_width = o_size.width * scale;
          cv::Size2f newsize(new_width, new_height);
          cv::RotatedRect temp_box(particles_centroid, newsize, rotation);
          cv::Rect2f estimated_box = temp_box.boundingRect();
          estimated_box = estimated_box & frame_window;

          jg::SearchArea search_area;
          cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
          mask(estimated_box).setTo(1);

          // Extract keypoints only inside the first estimated bounding box
          feature_extractor.UpdateFeatures(frame, mask, search_area.keypoints,
                                           search_area.descriptors);
          std::vector<cv::KeyPoint> matched_keypoints =
              match_finder.ComputeDescriptorMatches(
                  search_area.keypoints, search_area.descriptors,
                  target.original_keypoints, target.original_descriptors,
                  background_model.descriptors);

          // Fuse matched and tracked keypoints
          std::vector<cv::KeyPoint> matched_tracked_keypoints =
              jg::CombineKeypoints(matched_keypoints, tracked_keypoints_inside);

          jg::SortKeypointsByID(matched_tracked_keypoints);
          jg::EliminateDuplicatedIDKeypoints(matched_tracked_keypoints);

          // Eliminate outliers and estimate centroid
          std::vector<cv::KeyPoint> inliers = voting.FindConsensus(
              matched_tracked_keypoints, target.original_conditioned_keypoints,
              scale, rotation, centroid);
          matched_tracked_keypoints = inliers;

          if (!std::isnan(centroid.x) && !std::isnan(centroid.y)) {
            std::vector<std::vector<cv::DMatch>> all_matches =
                match_finder.ComputeAllDescriptorMatches(
                    search_area.descriptors, target.original_descriptors);

            std::vector<cv::KeyPoint>
                transformed_original_conditioned_keypoints =
                    jg::RotateAndScaleKeypoints(
                        target.original_conditioned_keypoints, rotation, scale);

            float descriptor_length = 512;
            std::vector<cv::KeyPoint> new_inliers =
                voting.DissambiguateCandidateKeypoints(
                    search_area.keypoints, all_matches, descriptor_length,
                    centroid, transformed_original_conditioned_keypoints);

            //          cv::Scalar blue = cv::Scalar(255, 0, 0);
            //          ot::Utils::DrawKeypoints(new_inliers, frame, blue,
            //                                   "Matched keypoints");
            jg::SortKeypointsByID(new_inliers);
            jg::SortKeypointsByID(matched_tracked_keypoints);

            std::vector<cv::KeyPoint> fused_keypoints =
                jg::CombineKeypoints(new_inliers, matched_tracked_keypoints);

            jg::EliminateDuplicatedIDKeypoints(fused_keypoints);

            inliers = voting.FindConsensus(
                fused_keypoints, target.original_conditioned_keypoints, scale,
                rotation, centroid);
            fused_keypoints = inliers;

            //            cv::Scalar green = cv::Scalar(0, 255, 0);
            //          jg::DrawKeypoints(fused_keypoints, frame, green, "Fused
            //          keypoints");
            jg::log() << "Number of fused points  " << fused_keypoints.size()
                      << "\n";

            target.active_keypoints = fused_keypoints;

            if (target.active_keypoints.size() >
                (target.original_keypoints.size() / 1.2)) {
              new_width = target.initial_size.width * scale;
              new_height = target.initial_size.height * scale;
              cv::Size2f newsize(new_width, new_height);
              cv::RotatedRect newbox =
                  cv::RotatedRect(centroid, newsize, (rotation / M_PI) * 180);
              cv::Rect2f nbox = newbox.boundingRect();
              cv::rectangle(temp_frame, nbox, cv::Scalar(0, 0, 255));
              target.ChangeBoundingBox(nbox);

            } else {
              target.Translate(particles_centroid, frame_window);
            }
          }
        } else {
          target.Translate(particles_centroid, frame_window);
        }

      } else {  // If particles accumulated weight is zero
        particle_filters->SpreadParticles();

        // Keypoints Tracking Module
        cv::Point2f centroid(nan, nan);
        float scale = nan;
        float rotation = nan;

        std::vector<cv::KeyPoint> tracked_keypoints =
            tracker.TrackKeypointsWithOpticalFlow(
                previous_frame_gray, frame_gray, target.active_keypoints);

        jg::SortKeypointsByID(tracked_keypoints);
        jg::EliminateDuplicatedIDKeypoints(tracked_keypoints);

        voting.EstimateScaleRotation(
            tracked_keypoints, target.original_distances_pairwise,
            target.original_angles_pairwise, scale, rotation);

        // Eliminate outliers and estimate centroid
        std::vector<cv::KeyPoint> inliers = voting.FindConsensus(
            tracked_keypoints, target.original_conditioned_keypoints, scale,
            rotation, centroid);
        tracked_keypoints = inliers;

        if (!std::isnan(centroid.x) && !std::isnan(centroid.y)) {
          float new_width = target.initial_size.width * scale;
          float new_height = target.initial_size.height * scale;
          cv::Size2f newsize(new_width, new_height);
          cv::Point2f tl(centroid.x - (new_width / 2),
                         centroid.y - (new_height / 2));
          cv::Rect2f area(tl, newsize);
          jg::SearchArea search_area;
          cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
          mask(area).setTo(1);
          feature_extractor.UpdateFeatures(frame, mask, search_area.keypoints,
                                           search_area.descriptors);

          std::vector<cv::KeyPoint> matched_keypoints =
              match_finder.ComputeDescriptorMatches(
                  search_area.keypoints, search_area.descriptors,
                  target.original_keypoints, target.original_descriptors,
                  background_model.descriptors);

          // Fuse matched and tracked keypoints
          std::vector<cv::KeyPoint> matched_tracked_keypoints =
              jg::CombineKeypoints(matched_keypoints, tracked_keypoints);

          jg::SortKeypointsByID(matched_tracked_keypoints);
          jg::EliminateDuplicatedIDKeypoints(matched_tracked_keypoints);

          std::vector<std::vector<cv::DMatch>> all_matches =
              match_finder.ComputeAllDescriptorMatches(
                  search_area.descriptors, target.original_descriptors);

          std::vector<cv::KeyPoint> transformed_original_conditioned_keypoints =
              jg::RotateAndScaleKeypoints(target.original_conditioned_keypoints,
                                          rotation, scale);

          float descriptor_length = 512;
          std::vector<cv::KeyPoint> new_inliers =
              voting.DissambiguateCandidateKeypoints(
                  search_area.keypoints, all_matches, descriptor_length,
                  centroid, transformed_original_conditioned_keypoints);

          jg::SortKeypointsByID(new_inliers);
          jg::SortKeypointsByID(matched_tracked_keypoints);

          std::vector<cv::KeyPoint> fused_keypoints =
              jg::CombineKeypoints(new_inliers, matched_tracked_keypoints);

          //            cv::Scalar green = cv::Scalar(0, 255, 0);
          //          jg::DrawKeypoints(fused_keypoints, frame, green, "Fused
          //          keypoints");
          jg::log() << "Number of fused points  " << fused_keypoints.size()
                    << "\n";

          target.active_keypoints = fused_keypoints;

          cv::rectangle(temp_frame, area, cv::Scalar(0, 0, 255));

          target.ChangeBoundingBox(area);
        }
      }

      //      jg::Mat3Uchar extra_frame_2;
      //      temp_frame.copyTo(extra_frame_2);
      //      particle_filters->PrintParticles(extra_frame_2);
      //      cv::imshow("Final Particles", extra_frame_2);

      //      cv::Scalar crazy = cv::Scalar(0, 255, 255);
      //      jg::DrawKeypoints(tracked_keypoints_inside, frame, crazy,
      //                        "Tracked keypoints inside");
      if (kShowImages) {
        manager.display()->ShowAllTargets(targets, temp_frame, "Target");
      }

      if (delay != -1) {
        cv::waitKey(delay);
      }

      jg::UpdateTracks(targets, frame_window, track);

      if (iter.hasNext()) {
        frame_gray.copyTo(previous_frame_gray);
        frame.copyTo(previous_frame);
        iter.next().copyTo(big_frame);
        cv::resize(big_frame, frame, cv::Size(), image_scale, image_scale,
                   cv::INTER_AREA);
        frame.copyTo(temp_frame);
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        ++frame_number;
      } else {
        can_continue = false;
      }
    } while (can_continue);

    // Save results for keypoint experiments
    auto folder = cmd_line.results_path + keypoint_extractor +
                  std::string("-") + keypoint_descriptor + std::string("/");

    if (!fs::is_directory(folder)) {
      mt::check(fs::create_directory(folder), "Cannot create folder '%s'",
                folder.c_str());
    }

    std::string tracks_path = folder + kMethodName + "-1.txt";

    int i = 1;
    while (fs::is_regular_file(tracks_path)) {
      tracks_path = folder + kMethodName + "-" + std::to_string(++i) + ".txt";
      if (i == 6) {
        return;
      }
    }


    // When testing different histograms
//    auto folder = cmd_line.results_path + histogram_type + std::string("/") +
//            color_space + std::string("/");

//    if (!fs::is_directory(folder)) {
//      mt::check(fs::create_directory(folder), "Cannot create folder '%s'",
//                folder.c_str());
//    }

//    std::string tracks_path = folder + kMethodName + "-1.txt";

//    int i = 1;
//    while (fs::is_regular_file(tracks_path)) {
//      tracks_path = folder + kMethodName + "-" + std::to_string(++i) + ".txt";
//      if (i == 6) {
//        return;
//      }
//    }

    // When testing a single combination of keypoints and
    // descriptors
//      std::string tracks_path = cmd_line.results_path + kMethodName +
//      "-1.txt";
//        int i = 1;
//        while (fs::is_regular_file(tracks_path)) {
//          tracks_path = cmd_line.results_path + kMethodName + "-" +
//                        std::to_string(++i) + ".txt";
//          if (i == 6) {
//            return;
//          }
//        }

    jg::SaveTracksInFile(track, tracks_path);
  }
}

int main(int argc, const char **argv) {
  if (argc < 2) {
    ShowHelpMessage();
    return EXIT_FAILURE;
  }

  try {
    const auto cmd_line = ReadCommandLine(argc, argv);

    if (cmd_line.command == HELP) {
      ShowHelpMessage();
      return EXIT_SUCCESS;
    }

    if (cmd_line.command == TRACK) {
      //      std::cout << "cv::getNumThreads() is " << cv::getNumThreads() <<
      //      '\n'
      //                << "cv::useOptimized()  is " << cv::useOptimized() <<
      //                std::endl;
      //      ProfilerStart("/tmp/jgthesis.prof");
      //      HeapProfilerStart("/tmp/objecttracker.prof");
      TrackWithParticleFilterAndKeypoints(cmd_line);
      //      HeapProfilerDump("/tmp/objecttracker.prof");
      //      HeapProfilerStop();
      //      ProfilerStop();
      return EXIT_SUCCESS;
    }

  } catch (CommandLine::Error &error) {
    std::cerr << "Invalid command line: " << error.what() << "\nTry "
              << TOOLNAME << ' ' << HELP << std::endl;

  } catch (std::exception &error) {
    std::cerr << error.what() << std::endl;
  }

  return EXIT_FAILURE;
}
