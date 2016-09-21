#include "jgtracker/SIRParticleFilter.h"

#include <iostream>
#include <math.h>
#include <random>
#include <boost/property_tree/ini_parser.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui.hpp>
#include "jgtracker/operations.h"
#include "jgtracker/types.h"
#include "jgtracker/thirdparty/mt/assert.h"
#include "jgtracker/thirdparty/mt/check.h"

namespace jg {

namespace {

const cv::Vec3b kParticlesColor(200, 150, 0);
const float kZero = 0;

}  // namespace

SIRParticleFilter::SIRParticleFilter(Target& target, size_t num_particles,
                                     size_t num_stripes)
    : target_id(target.id) {
  MT_ASSERT_GT(num_particles, 0);
  target_p = &target;

  mt::check(target.HasHistogram(), "The target's does not have an histogram");
  Particle particle;
  particle.x_p = target.bounding_box.x;
  particle.y_p = target.bounding_box.y;
  particle.box = target.bounding_box;
  particle.weight = 0;

  particles = std::vector<Particle>(num_particles, particle);

  target_height = target.bounding_box.height;
  target_width = target.bounding_box.width;

  stripes = CreateImageStripes(target_width, target_height, num_stripes);
}

cv::Point2f SIRParticleFilter::ComputeParticlesCentroid(float image_width,
                                                        float image_height) {
  float last_col = image_width - 1;
  float last_row = image_height - 1;
  cv::Point2f tl(0, 0);

  for (const Particle& p : particles) {
    float weight = p.weight;
    tl.x += weight * std::max(kZero, std::min(p.box.x, last_col));
    tl.y += weight * std::max(kZero, std::min(p.box.y, last_row));
  }
  tl = cv::Point2f(std::floor(tl.x), std::floor(tl.y));
  cv::Point2f half_box(std::floor(target_width * 0.5),
                       std::floor(target_height * 0.5));
  cv::Point2f centroid = tl + half_box;
  return centroid;
}

cv::Rect2f SIRParticleFilter::ComputeParticlesEnclosingRec() const {
  float x_min = std::numeric_limits<float>::max();
  float x_max = 0;
  float y_min = std::numeric_limits<float>::max();
  float y_max = 0;

  for (const Particle& p : particles) {
    x_min = std::min(x_min, p.box.x);
    x_max = std::max(x_max, p.box.br().x);
    y_min = std::min(y_min, p.box.y);
    y_max = std::max(y_max, p.box.br().y);
  }

  cv::Point2f tl(x_min, y_min);
  cv::Point2f br(x_max, y_max);

  cv::Rect2f rec(tl, br);
  MT_ASSERT_GT(rec.area(), 0);
  return rec;
}

//------------------------------------------------------------------------------
// SIRParticleFilters
//------------------------------------------------------------------------------

SIRParticleFilters::SIRParticleFilters(const std::string& config_file,
                                       const Mat3Uchar& frame) {
  frame_window_ = cv::Rect2f(0.0, 0.0, frame.cols, frame.rows);

  boost::property_tree::ptree config;
  boost::property_tree::ini_parser::read_ini(config_file.c_str(), config);
  num_particles_ = config.get<size_t>("ParticleFilter.num_particles");
  x_sigma_ = config.get<float>("ParticleFilter.x_sigma");
  y_sigma_ = config.get<float>("ParticleFilter.y_sigma");
  a1_ = config.get<float>("ParticleFilter.a1");
  a2_ = config.get<float>("ParticleFilter.a2");
  lambda_ = config.get<int>("ParticleFilter.lambda");
  num_stripes_ = config.get<size_t>("ParticleFilter.num_stripes");
}

void SIRParticleFilters::CreateFilter(Target& target) {
  bool target_has_filter = false;
  for (const SIRParticleFilter& filter : filters_) {
    if (target.id == filter.target_id) {
      target_has_filter = true;
      break;
    }
  }
  if (!target_has_filter) {
    filters_.emplace_back(target, num_particles_, num_stripes_);
  }
}

void SIRParticleFilters::DriftAndDiffuse() {
  std::mt19937 gen(rd_());
  std::normal_distribution<> x_rnd(0, x_sigma_);
  std::normal_distribution<> y_rnd(0, y_sigma_);

  for (SIRParticleFilter& filter : filters_) {
    for (Particle& particle : filter.particles) {
      while (true) {
        double x = a1_ * particle.box.x + a2_ * particle.x_p + x_rnd(gen);
        double y = a1_ * particle.box.y + a2_ * particle.y_p + y_rnd(gen);
        particle.x_p = particle.box.x;
        particle.y_p = particle.box.y;
        cv::Rect2f box(cv::Point2f(std::floor(x), std::floor(y)),
                       cv::Size2f(filter.target_width, filter.target_height));
        box = box & frame_window_;
        if (box.area() > 0) {
          particle.box = box;
          break;
        }
      }
    }
  }
}

void SIRParticleFilters::SpreadParticles() {
  std::mt19937 gen(rd_());
  std::uniform_real_distribution<> x_rnd(0, frame_window_.width);
  std::uniform_real_distribution<> y_rnd(0, frame_window_.height);

  for (SIRParticleFilter& filter : filters_) {
    float uniform_weight;
    if (!filter.particles.empty()) {
      uniform_weight = 1 / filter.particles.size();
    }
    for (Particle& particle : filter.particles) {
      while (true) {
        double x = x_rnd(gen);
        double y = y_rnd(gen);
        particle.x_p = (x - a1_ * particle.box.x) / -a2_;
        particle.y_p = (y - particle.box.y) / -a2_;
        cv::Rect2f box(cv::Point2f(std::floor(x), std::floor(y)),
                       cv::Size2f(filter.target_width, filter.target_height));
        box = box & frame_window_;
        if (box.area() > 0) {
          particle.box = box;
          particle.weight = uniform_weight;
          break;
        }
      }
    }
  }
}

void SIRParticleFilters::PrintParticles(Mat3Uchar& frame) const {
  for (const SIRParticleFilter& filter : filters_) {
    for (const Particle& particle : filter.particles) {
      cv::Point2f center(
          particle.box.x + std::floor(particle.box.width * 0.5),
          particle.box.y + std::floor(particle.box.height * 0.5));
      size_t x = static_cast<size_t>(
          std::max(kZero, std::min(center.x, frame_window_.width - 1)));
      size_t y = static_cast<size_t>(
          std::max(kZero, std::min(center.y, frame_window_.height - 1)));
      cv::circle(frame, cv::Point2f(x, y), 1, kParticlesColor, 1, cv::LINE_8);
    }
  }
}

void SIRParticleFilters::PrintParticlesWindows(Mat3Uchar& frame,
                                               const cv::Scalar& color) const {
  for (const SIRParticleFilter& filter : filters_) {
    for (const Particle& particle : filter.particles) {
      cv::rectangle(frame, particle.box, color);
    }
  }
}

// Douc, R. and Cappe, O. and Moulines, E., “Comparison of Resampling
// Schemes for Particle Filtering”, Image and Signal Processing and Analysis,
// 2005. Residual sampling
void SIRParticleFilters::ResampleParticles() {
  std::mt19937 gen(rd_());
  std::uniform_real_distribution<> distribution(0, 1);

  for (SIRParticleFilter& filter : filters_) {
    std::vector<float> weights;
    std::vector<float> residual_counts;
    for (size_t i = 0; i != num_particles_; i++) {
      weights.push_back(num_particles_ * filter.particles.at(i).weight);
      residual_counts.push_back(std::floor(weights.back()));
    }

    auto residuals_sum =
        std::accumulate(residual_counts.begin(), residual_counts.end(), 0.0);
    size_t remainder = num_particles_ - static_cast<size_t>(residuals_sum);

    std::vector<double> new_weights;

    for (size_t i = 0; i != num_particles_; i++) {
      new_weights.push_back((weights.at(i) - residual_counts.at(i)) /
                            static_cast<double>(remainder));
    }

    std::vector<size_t> indexes;
    for (size_t j = 0; j != num_particles_; j++) {
      for (size_t k = 0; k != residual_counts.at(j); k++) {
        indexes.push_back(j);
      }
    }

    std::vector<double> q_weights(new_weights.size(), 0.0);
    std::partial_sum(new_weights.begin(), new_weights.end(), q_weights.begin());
    q_weights.back() = 1;
    for (size_t i = indexes.size(); i != num_particles_; i++) {
      auto random_num = distribution(gen);
      size_t j = 0;
      while (q_weights.at(j) < random_num) {
        j++;
      }
      indexes.push_back(j);
    }
    std::vector<Particle> resampled_particles;
    for (size_t i = 0; i != num_particles_; i++) {
      resampled_particles.push_back(filter.particles.at(indexes.at(i)));
    }
    filter.particles = resampled_particles;

    // Set uniform weights
    MT_ASSERT_GT(num_particles_, 0);
    const double weight = 1.0 / num_particles_;
    for (Particle& particle : filter.particles) {
      particle.weight = weight;
    }
    log() << "Resample! \n\n";
  }
}

void SIRParticleFilters::UpdateFilters(std::vector<Target>& targets) {
  if (!filters_.empty()) {
    // Check if there are new targets
    for (Target& target : targets) {
      bool target_has_filter = false;
      for (const SIRParticleFilter& filter : filters_) {
        if (target.id == filter.target_id) {
          target_has_filter = true;
          break;
        }
      }
      if (!target_has_filter) {
        filters_.emplace_back(target, num_particles_, num_stripes_);
      }
    }

    // Eliminate filters from no longer existing targets
    std::vector<SIRParticleFilter> new_filters;
    while (!filters_.empty()) {
      bool last_filter_target_exists = false;
      for (auto& target : targets) {
        if (target.id == filters_.back().target_id) {
          last_filter_target_exists = true;
          filters_.back().target_p = &target;
          break;
        }
      }
      if (last_filter_target_exists) {
        new_filters.push_back(filters_.back());
      }
      filters_.pop_back();
    }
    if (!new_filters.empty()) {
      filters_ = new_filters;
    }
  } else {
    for (Target& target : targets) {
      filters_.emplace_back(target, num_particles_, num_stripes_);
    }
  }
}

//------------------------------------------------------------------------------
// SIRParticleFiltersNormalHistogram
//------------------------------------------------------------------------------
void SIRParticleFiltersNormalHistogram::SetupColorSpace(
    ColorSpaceEnum color_space_enum) {
  switch (color_space_enum) {
    case kHSV:
      histogram_creator_.reset(new HsvHistogramFactory);
      break;
    default:
      histogram_creator_.reset(new GrayscaleHistogramFactory);
      break;
  }
}

void SIRParticleFiltersNormalHistogram::ComputeNormalizedWeights(
    const cv::Mat& frame) {
  for (SIRParticleFilter& filter : filters_) {
    mt::check(filter.target_p->HasHistogram(),
              "The target's does not include histogram");
    auto original_hist =
        filter.target_p->GetOriginalHistogram()->histogram_matrix();

    filter.accumulated_weight = 0.0;
    for (Particle& particle : filter.particles) {
      cv::Rect2f box = particle.box & frame_window_;
      if (box.area() > 1) {
        cv::Mat roi_mask(frame.size(), CV_8UC1, cv::Scalar::all(0));
        roi_mask(box).setTo(cv::Scalar::all(255));
        cv::Mat particle_hist;

        histogram_creator_->Create(frame, roi_mask, particle_hist);
        MT_ASSERT_FALSE(particle_hist.empty());
        cv::normalize(particle_hist, particle_hist, 0, 255, cv::NORM_MINMAX);

        double dist_to_original = cv::compareHist(particle_hist, original_hist,
                                                  cv::HISTCMP_HELLINGER);        

        MT_ASSERT_GE(dist_to_original, 0);
        MT_ASSERT_LE(dist_to_original, 1);
        float dist = static_cast<float>(dist_to_original);

        particle.weight = std::exp(-lambda_ * dist);
        MT_ASSERT_GE(particle.weight, 0);
        filter.accumulated_weight += particle.weight;
      } else {
        particle.weight = 0;
      }
    }
    // Normalize weights
    if (filter.accumulated_weight > 0) {
      for (Particle& particle : filter.particles) {
        particle.weight /= filter.accumulated_weight;
      }
    }
  }
}

//------------------------------------------------------------------------------
// SIRParticleFiltersIntegralHistogram
//------------------------------------------------------------------------------
void SIRParticleFiltersIntegralHistogram::SetupColorSpace(
    ColorSpaceEnum color_space_enum) {
  switch (color_space_enum) {
    case kHSV:
      integral_histogram_creator_.reset(new HsvIntegralHistogramFactory);
      break;
    default:
      integral_histogram_creator_.reset(new GrayscaleIntegralHistogramFactory);
      break;
  }
}

// This version computes subparts histograms

void SIRParticleFiltersIntegralHistogram::ComputeNormalizedWeights(
    const cv::Mat& frame) {
  for (SIRParticleFilter& filter : filters_) {
    log() << "Start: ComputeNormalizedWeights2\n";

    log() << "B : ComputeNormalizedWeights2\n";

    cv::Rect2f particles_rec = filter.ComputeParticlesEnclosingRec();
    log() << "B : ComputeNormalizedWeights2\n";
    particles_rec = particles_rec & frame_window_;
    cv::Mat particles_region;
    frame(particles_rec).copyTo(particles_region);

    log() << "Region rows: " << particles_region.rows << "\n";
    log() << "Region cols: " << particles_region.cols << "\n";

    // a,b,c are the channels of the selected color space
    // For HSV, a = h and b = s
    // For Grayscale, there is only channel a
    std::vector<cv::Mat> channel_a_particle;
    std::vector<cv::Mat> channel_b_particle;
    std::vector<cv::Mat> channel_c_particle;
    log() << "C: ComputeNormalizedWeights2\n";

    integral_histogram_creator_->ComputeIntegralBinaryMasks(
        particles_region, channel_a_particle, channel_b_particle,
        channel_c_particle);

    log() << "D: ComputeNormalizedWeights2\n";

    filter.accumulated_weight = 0.0;

//    cv::imshow("Particles region", particles_region);
    //    cvWaitKey();


    for (Particle& particle : filter.particles) {
      cv::Rect2f box = particle.box;
      box = box & frame_window_;
      if (box.area() > 1) {
        // Condition the coordinates of the particle's bounding box
        // since the integral histogram was build only with the image region
        // that encloses all the particles
        // From image coordinates to particles rect coordinates
        cv::Rect2f new_box = box - particles_rec.tl();

        MT_ASSERT_LE(new_box.br().x, particles_rec.width);
        MT_ASSERT_LE(new_box.br().y, particles_rec.height);
        MT_ASSERT_GE(new_box.tl().x, 0);
        MT_ASSERT_GE(new_box.tl().y, 0);

        cv::rectangle(particles_region, new_box, cv::Scalar(255, 255, 255));
        //      cv::imshow("Particles region", particles_region);
        //      cvWaitKey(0);
        cv::Rect2f particles_rec_frame(
            cv::Point2f(0, 0),
            cv::Size2f(particles_rec.width, particles_rec.height));

        float parts_sum = 0;
        size_t stripe_index = 0;
        for (const cv::Rect2f& stripe : filter.stripes) {
          cv::Rect2f temp_stripe = stripe + new_box.tl();
          cv::Rect2f shifted_stripe = temp_stripe & particles_rec_frame;

          if (shifted_stripe.area() > 0) {
            MT_ASSERT_LE(shifted_stripe.br().x, particles_rec.width);
            MT_ASSERT_LE(shifted_stripe.br().y, particles_rec.height);
            MT_ASSERT_GE(shifted_stripe.tl().x, 0);
            MT_ASSERT_GE(shifted_stripe.tl().y, 0);

            cv::rectangle(particles_region, shifted_stripe,
                          cv::Scalar(0, 0, 0));

            std::vector<float> histogram_vector =
                integral_histogram_creator_->CreateHistogramVector(
                    shifted_stripe, channel_a_particle, channel_b_particle,
                    channel_c_particle);

            const std::vector<float>& original_histogram_vector =
                filter.target_p->stripes_histograms.at(stripe_index++);

            MT_ASSERT_EQ(original_histogram_vector.size(),
                         histogram_vector.size());

            float vector_prod_sum = 0;
            for (size_t i = 0; i != histogram_vector.size(); i++) {
              vector_prod_sum += std::sqrt(histogram_vector.at(i) *
                                           original_histogram_vector.at(i));
            }

            parts_sum += 1 - vector_prod_sum;
          }
        }
        particle.weight = std::exp(-lambda_ * parts_sum);
        filter.accumulated_weight += particle.weight;

      } else {
        // box.area() < 1
        particle.weight = 0;
      }
    }
    // Normalize weights
    if (filter.accumulated_weight > 0) {
      for (Particle& particle : filter.particles) {
        particle.weight /= filter.accumulated_weight;
      }
    }
  }
}

}  // namespace jg
