#include "IntegralHistogramFactory.h"

#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"
#include <opencv2/imgproc.hpp>

#include "jgtracker/operations.h"
#include "jgtracker/types.h"

namespace jg {

void GrayscaleIntegralHistogramFactory::ComputeIntegralBinaryMasks(
    const cv::Mat &image, std::vector<cv::Mat> &grayscale_mats,
    std::vector<cv::Mat> & /*b_mats*/, std::vector<cv::Mat> & /*c_mats*/) {
  log() << "Start: ComputeGrayscaleIntegralMatrices\n";
  grayscale_mats.clear();

  cv::Mat grayscale_image;
  cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);

  float num_g_bins = 32;
  float g_max_value = 255;
  size_t num_layers = static_cast<size_t>(num_g_bins);
  cv::Mat mask_g[num_layers];

  mask_g[0] = cv::Mat::ones(grayscale_image.rows, grayscale_image.cols, CV_8U);
  for (size_t i = 1; i != num_layers; i++) {
    float threshold = (i * g_max_value / num_g_bins);
    cv::threshold(grayscale_image, mask_g[i], threshold, g_max_value,
                  cv::THRESH_BINARY);
  }

  for (size_t i = 0; i != num_layers - 1; i++) {
    cv::Mat xor_mat;
    cv::bitwise_xor(mask_g[i], mask_g[i + 1], xor_mat);
    xor_mat.convertTo(xor_mat, CV_32F);
    grayscale_mats.push_back(
        cv::Mat(xor_mat.rows + 1, xor_mat.cols + 1, CV_32F));
    cv::integral(xor_mat, grayscale_mats.back(), CV_32F);
  }

  grayscale_mats.push_back(cv::Mat(0, 0, CV_32F));
  cv::integral(mask_g[num_layers - 1], grayscale_mats.back(), CV_32F);

  log() << "Finish: ComputeGrayscaleIntegralMatrices\n";
}

std::vector<float> GrayscaleIntegralHistogramFactory::CreateHistogramVector(
    const cv::Rect2f &region, const std::vector<cv::Mat> &grayscale_mats,
    const std::vector<cv::Mat> & /*b_mats*/,
    const std::vector<cv::Mat> & /*c_mats*/) {
  cv::Point2f tl = region.tl();
  cv::Point2f br = region.br();

  MT_ASSERT_LT(br.x, grayscale_mats[0].cols);
  MT_ASSERT_LT(br.y, grayscale_mats[0].rows);
  MT_ASSERT_GE(tl.x, 0);
  MT_ASSERT_GE(tl.y, 0);

  std::vector<float> grayscale_vector;
  for (size_t i = 0; i != grayscale_mats.size(); i++) {
    const cv::Mat &g_mat = grayscale_mats.at(i);
    MT_ASSERT_EQ(g_mat.type(), CV_32F);
    float tl_sum = g_mat.at<float>(tl.y, tl.x);
    float tr_sum = g_mat.at<float>(tl.y, br.x);
    float bl_sum = g_mat.at<float>(br.y, tl.x);
    float br_sum = g_mat.at<float>(br.y, br.x);
    grayscale_vector.push_back(br_sum - bl_sum - tr_sum + tl_sum);
  }

  float sum =
      std::accumulate(grayscale_vector.begin(), grayscale_vector.end(), 0.0);
  if (sum != 0) {
    for (size_t i = 0; i != grayscale_vector.size(); i++) {
      grayscale_vector.at(i) /= sum;
    }
  }

  return grayscale_vector;
}

void HsvIntegralHistogramFactory::ComputeIntegralBinaryMasks(
    const cv::Mat &image, std::vector<cv::Mat> &h_mats,
    std::vector<cv::Mat> &s_mats, std::vector<cv::Mat> & /*c_mats*/) {
  log() << "Start: ComputeHSIntegralMatrices\n";
  h_mats.clear();
  s_mats.clear();

  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

  cv::Mat channels[3];
  cv::split(hsv_image, channels);
  cv::Mat h_channel;
  channels[0].convertTo(h_channel, CV_32F);
  cv::Mat s_channel;
  channels[1].convertTo(s_channel, CV_32F);
  cv::Mat v_channel;
  channels[2].convertTo(v_channel, CV_32F);

  // Compute H channel vector component
  float num_h_bins = 10;  // 30
  float h_max_value = 179;
  const float max_value = 1;
  size_t num_layers = static_cast<size_t>(num_h_bins);
  cv::Mat mask[num_layers];

  mask[0] = cv::Mat::ones(h_channel.rows, h_channel.cols, CV_32F);
  for (size_t i = 1; i != num_layers; i++) {
    float threshold = (i * h_max_value / num_h_bins);
    cv::threshold(h_channel, mask[i], threshold, max_value, cv::THRESH_BINARY);
  }

  for (size_t i = 0; i != num_layers - 1; i++) {
    cv::Mat xor_mat;
    cv::bitwise_xor(mask[i], mask[i + 1], xor_mat);
    xor_mat.convertTo(xor_mat, CV_32F);
    h_mats.push_back(cv::Mat(xor_mat.rows + 1, xor_mat.cols + 1, CV_32F));
    cv::integral(xor_mat, h_mats.back(), CV_32F);
  }

  h_mats.push_back(cv::Mat(0, 0, CV_32F));
  cv::integral(mask[num_layers - 1], h_mats.back(), CV_32F);

  // Compute S channel vector component
  float num_s_bins = 10;  // 32
  float s_max_value = 255;
  num_layers = static_cast<size_t>(num_s_bins);
  cv::Mat mask_s[num_layers];

  for (size_t i = 0; i != num_layers; i++) {
    float threshold = (i * s_max_value / num_s_bins);
    cv::threshold(s_channel, mask_s[i], threshold, max_value,
                  cv::THRESH_BINARY);
  }

  for (size_t i = 0; i != num_layers - 1; i++) {
    cv::Mat xor_mat;
    cv::bitwise_xor(mask_s[i], mask_s[i + 1], xor_mat);
    xor_mat.convertTo(xor_mat, CV_32F);
    s_mats.push_back(cv::Mat(xor_mat.rows + 1, xor_mat.cols + 1, CV_32F));
    cv::integral(xor_mat, s_mats.back(), CV_32F);
  }

  s_mats.push_back(cv::Mat(0, 0, CV_32F));
  cv::integral(mask_s[num_layers - 1], s_mats.back(), CV_32F);

  //  // Compute V channel vector component
  //  float num_v_bins = 10;  // 32
  //  float v_max_value = 255;
  //  num_layers = static_cast<size_t>(num_v_bins);
  //  cv::Mat mask_v[num_layers];
  //  for (size_t i = 0; i != num_layers; i++) {
  //    float threshold = (i * v_max_value / num_v_bins);
  //    cv::threshold(v_channel, mask_v[i], threshold, max_value,
  //                  cv::THRESH_BINARY);
  //  }

  //  for (size_t i = 0; i != num_layers - 1; i++) {
  //    cv::Mat xor_mat;
  //    cv::bitwise_xor(mask_v[i], mask_v[i + 1], xor_mat);
  //    xor_mat.convertTo(xor_mat, CV_32F);
  //    v_mats.push_back(cv::Mat(xor_mat.rows + 1, xor_mat.cols + 1, CV_32F));
  //    cv::integral(xor_mat, v_mats.back(), CV_32F);
  //  }

  //  v_mats.push_back(cv::Mat(0, 0, CV_32F));
  //  cv::integral(mask_v[num_layers - 1], v_mats.back(), CV_32F);

  log() << "Finish: ComputeHSIntegralMatrices";
}

std::vector<float> HsvIntegralHistogramFactory::CreateHistogramVector(
    const cv::Rect2f &region, const std::vector<cv::Mat> &h_mats,
    const std::vector<cv::Mat> &s_mats,
    const std::vector<cv::Mat> & /*c_mats*/) {
  //  cv::Point2f tl = region.tl();
  //  cv::Point2f br(region.br().x -1, region.br().y -1);
  cv::Point2f tl = region.tl();
  cv::Point2f br(std::min(region.br().x - 1, static_cast<float>(h_mats[0].cols)),
                 std::min(region.br().y - 1, static_cast<float>(h_mats[0].rows)));

  MT_ASSERT_LT(br.x, h_mats[0].cols);
  MT_ASSERT_LT(br.y, h_mats[0].rows);
  MT_ASSERT_GE(tl.x, 0);
  MT_ASSERT_GE(tl.y, 0);

  std::vector<float> h_vector;
  for (size_t i = 0; i != h_mats.size(); i++) {
    const cv::Mat &h_mat = h_mats.at(i);
    MT_ASSERT_EQ(h_mat.type(), CV_32F);
    float tl_sum = h_mat.at<float>(tl.y, tl.x);
    float tr_sum = h_mat.at<float>(tl.y, br.x);
    float bl_sum = h_mat.at<float>(br.y, tl.x);
    float br_sum = h_mat.at<float>(br.y, br.x);
    h_vector.push_back(br_sum - bl_sum - tr_sum + tl_sum);
  }

  float sum = std::accumulate(h_vector.begin(), h_vector.end(), 0.0);
  if (sum != 0) {
    for (size_t i = 0; i != h_vector.size(); i++) {
      h_vector.at(i) /= sum;
    }
  }

  std::vector<float> s_vector;
  for (size_t i = 0; i != s_mats.size(); i++) {
    const cv::Mat &s_mat = s_mats.at(i);
    MT_ASSERT_EQ(s_mat.type(), CV_32F);
    float tl_sum = s_mat.at<float>(tl.y, tl.x);
    float tr_sum = s_mat.at<float>(tl.y, br.x);
    float bl_sum = s_mat.at<float>(br.y, tl.x);
    float br_sum = s_mat.at<float>(br.y, br.x);
    s_vector.push_back(br_sum - bl_sum - tr_sum + tl_sum);
  }

  sum = std::accumulate(s_vector.begin(), s_vector.end(), 0.0);
  if (sum != 0) {
    for (size_t i = 0; i != s_vector.size(); i++) {
      s_vector.at(i) /= sum;
    }
  }

  std::vector<float> hs_vector;
  hs_vector.reserve(h_vector.size() + s_vector.size());
  hs_vector.insert(hs_vector.end(), h_vector.begin(), h_vector.end());
  hs_vector.insert(hs_vector.end(), s_vector.begin(), s_vector.end());

  sum = std::accumulate(hs_vector.begin(), hs_vector.end(), 0.0);
  if (sum != 0) {
    for (size_t i = 0; i != hs_vector.size(); i++) {
      hs_vector.at(i) /= sum;
    }
  }

  return hs_vector;
}

}  // namespace jg
