#include "FeaturesExtractor.h"

#include <boost/property_tree/ini_parser.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "jgtracker/operations.h"
#include "jgtracker/thirdparty/mt/assert.h"

namespace jg {

namespace {

const int kNumNearestNeighbours = 2;
const double kRatio = 0.4;  // 0.8

}  // namespace

FeaturesExtractor::FeaturesExtractor(
    KeypointDetectorEnum keypoint_detector_enum,
    KeypointDescriptorEnum keypoint_descriptor_enum,
    const std::string &config_file) {
  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);

  switch (keypoint_detector_enum) {
    case kBRISKDetector:
      detector_p_ =
          cv::BRISK::create(config.get<std::int64_t>("BRISK.threshold"),
                            config.get<std::int64_t>("BRISK.octaves"),
                            config.get<std::double_t>("BRISK.pattern_scale"));
      break;
    case kFASTDetector:
      detector_p_ = cv::FastFeatureDetector::create();
      break;
    case kORBDetector:
      detector_p_ =
          cv::ORB::create(config.get<std::int64_t>("ORB.n_features"),
                          config.get<std::double_t>("ORB.scale_factor"),
                          config.get<std::int64_t>("ORB.n_levels"),
                          config.get<std::int64_t>("ORB.edge_threshold"),
                          config.get<std::int64_t>("ORB.first_level"),
                          config.get<std::int64_t>("ORB.WTA_K"),
                          config.get<std::int64_t>("ORB.scoreType"),
                          config.get<std::int64_t>("ORB.patch_size"),
                          config.get<std::int64_t>("ORB.fast_threshold"));
      break;
    case kSIFTDetector:
      detector_p_ = cv::xfeatures2d::SIFT::create(
          config.get<std::int64_t>("SIFT.n_features"),
          config.get<std::int64_t>("SIFT.n_octave_layers"),
          config.get<std::double_t>("SIFT.contrast_threshold"),
          config.get<std::double_t>("SIFT.edge_threshold"),
          config.get<std::double_t>("SIFT.sigma"));
      break;
    default:
      detector_p_ = cv::xfeatures2d::SURF::create(
          config.get<std::int64_t>("SURF.hessian_threshold"),
          config.get<std::int64_t>("SURF.n_octaves"),
          config.get<std::int64_t>("SURF.n_octave_layers"),
          config.get<bool>("SURF.extended"), config.get<bool>("SURF.upright"));
      break;
  }

  switch (keypoint_descriptor_enum) {
    case kBRISKDescriptor:
      descriptor_p_ =
          cv::BRISK::create(config.get<std::int64_t>("BRISK.threshold"),
                            config.get<std::int64_t>("BRISK.octaves"),
                            config.get<std::double_t>("BRISK.pattern_scale"));
      break;
    case kORBDescriptor:
      descriptor_p_ =
          cv::ORB::create(config.get<std::int64_t>("ORB.n_features"),
                          config.get<std::double_t>("ORB.scale_factor"),
                          config.get<std::int64_t>("ORB.n_levels"),
                          config.get<std::int64_t>("ORB.edge_threshold"),
                          config.get<std::int64_t>("ORB.first_level"),
                          config.get<std::int64_t>("ORB.WTA_K"),
                          config.get<std::int64_t>("ORB.score_type"),
                          config.get<std::int64_t>("ORB.patch_size"),
                          config.get<std::int64_t>("ORB.fast_threshold"));
      break;
    case kSIFTDescriptor:
      descriptor_p_ = cv::xfeatures2d::SIFT::create(
          config.get<int>("SIFT.n_features"),
          config.get<int>("SIFT.n_octave_layers"),
          config.get<double>("SIFT.contrast_threshold"),
          config.get<double>("SIFT.edge_threshold"),
          config.get<double>("SIFT.sigma"));
      break;
    default:
      descriptor_p_ = cv::xfeatures2d::SURF::create(
          config.get<int>("SURF.hessian_threshold"),
          config.get<int>("SURF.n_octaves"),
          config.get<int>("SURF.n_octave_layers"),
          config.get<bool>("SURF.extended"), config.get<bool>("SURF.upright"));
      break;
  }
  descriptor_lenght_ = descriptor_p_->descriptorSize();
}

void FeaturesExtractor::UpdateFeatures(const Mat3Uchar &image,
                                       const cv::Rect2d &bounding_box,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       cv::Mat &descriptors) {
  log() << "Started updating features inside bounding box\n";

  MT_ASSERT_GT(bounding_box.area(), 0);

  cv::Mat detection_gray;
  cv::cvtColor(image, detection_gray, CV_BGR2GRAY);

  cv::Mat roi_mask(image.size(), CV_8U, cv::Scalar::all(0));
  roi_mask(bounding_box).setTo(cv::Scalar::all(255));
  MT_ASSERT_EQ(roi_mask.type(), CV_8U);
  MT_ASSERT_EQ(roi_mask.size(), detection_gray.size());

  keypoints.clear();
  detector_p_->detect(detection_gray, keypoints, roi_mask);

  size_t num_keypoints = keypoints.size();

  // Eliminate identical keypoints
  std::sort(keypoints.begin(), keypoints.end(),
            [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
              return a.pt.dot(a.pt) < b.pt.dot(b.pt);
            });

  auto last = std::unique(keypoints.begin(), keypoints.end(),
                          [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                            return (a.pt.x == b.pt.x) && (a.pt.y == b.pt.y);
                          });

  keypoints.erase(last, keypoints.end());

  size_t num_deleted_keypoints = num_keypoints - keypoints.size();

  log() << "Num deleted keypoints " << num_deleted_keypoints << "\n";

  descriptor_p_->compute(detection_gray, keypoints, descriptors);

  log() << "Finished updating features inside bounding box\n";
}

void FeaturesExtractor::UpdateFeatures(const Mat3Uchar &image,
                                       const cv::Mat &roi_mask,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       cv::Mat &descriptors) {
  log() << "Started updating features in roi\n";

  cv::Mat detection_gray;
  cv::cvtColor(image, detection_gray, CV_BGR2GRAY);

  std::vector<cv::KeyPoint> initial_keypoints;
  detector_p_->detect(detection_gray, initial_keypoints, roi_mask);

  // Eliminate identical keypoints
  std::sort(initial_keypoints.begin(), initial_keypoints.end(),
            [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
              return a.pt.dot(a.pt) < b.pt.dot(b.pt);
            });

  auto last = std::unique(initial_keypoints.begin(), initial_keypoints.end(),
                          [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                            return (a.pt.x == b.pt.x) && (a.pt.y == b.pt.y);
                          });

  initial_keypoints.erase(last, initial_keypoints.end());
  keypoints = initial_keypoints;

  descriptor_p_->compute(detection_gray, keypoints, descriptors);

  log() << "Finished updating features 2d in roi \n";

  //      cv::Mat output_image;
  //      cv::drawKeypoints(image, keypoints, output_image);
  //      cv::imshow("UpdateFeatures", output_image);
  //      cv::waitKey(0);
}

}  // namespace jg
