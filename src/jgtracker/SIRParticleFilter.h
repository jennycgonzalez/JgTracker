//     ___     _____              _
//    |_  |   |_   _|            | |
//      | | __ _| |_ __ __ _  ___| | _____ _ __
//      | |/ _` | | '__/ _` |/ __| |/ / _ \ '__|
//  /\__/ / (_| | | | | (_| | (__|   <  __/ |
//  \____/ \__, \_/_|  \__,_|\___|_|\_\___|_|
//         __/ |
//        |___/
//
// https://github.com/jennycgonzalez/jgtracker
//
// BSD 2-Clause License

/*
Copyright (c) 2016, Jenny Gonzalez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef JG_SIR_PARTICLE_FILTER_H
#define JG_SIR_PARTICLE_FILTER_H

#include "jgtracker/HistogramFactory.h"
#include "jgtracker/IntegralHistogramFactory.h"
#include "jgtracker/types.h"
#include "jgtracker/Target.h"

namespace jg {

struct Particle {
  float x_p = 0.0;  // previous tl x coordinate
  float y_p = 0.0;  // previous tl y coordinate
  float weight = 0;
  cv::Rect2f box;

  Particle() = default;
  Particle(const Particle&) = default;
};

struct SIRParticleFilter {
  size_t target_id;
  Target* target_p;
  float target_height = 0.0;
  float target_width = 0.0;
  std::vector<Particle> particles;
  float accumulated_weight = 100;
  cv::Point2f particles_centroid;
  std::vector<cv::Rect2f> stripes;

  SIRParticleFilter() = default;
  SIRParticleFilter(const SIRParticleFilter&) = default;
  SIRParticleFilter(SIRParticleFilter&&) = default;
  SIRParticleFilter& operator=(SIRParticleFilter&&) = default;
  SIRParticleFilter& operator=(const SIRParticleFilter&) = default;

  SIRParticleFilter(Target& target, size_t num_particles, size_t num_stripes);
  cv::Point2f ComputeParticlesCentroid(float image_width, float image_height);
  cv::Rect2f ComputeParticlesEnclosingRec() const;
};

//------------------------------------------------------------------------------
// SIRParticleFilters
//------------------------------------------------------------------------------

class SIRParticleFilters {
 public:
  std::vector<SIRParticleFilter> filters_;

  virtual ~SIRParticleFilters() = default;

  SIRParticleFilters(const std::string& config_file, const Mat3Uchar& frame);

  virtual void SetupColorSpace(ColorSpaceEnum color_space_enum) = 0;
  virtual void ComputeNormalizedWeights(const cv::Mat& frame) = 0;

  void CreateFilter(Target& target);
  void DriftAndDiffuse();
  void PrintParticles(Mat3Uchar& frame) const;
  void PrintParticlesWindows(Mat3Uchar& frame, const cv::Scalar& color) const;
  void ResampleParticles();
  void SpreadParticles();
  void UpdateFilters(std::vector<Target>& targets);
  SIRParticleFilter& GetFilter(size_t index) { return filters_.at(index); }

 protected:
  std::random_device rd_;
  size_t num_particles_ = 300;
  double x_sigma_ = 1.0;
  double y_sigma_ = 1.0;
  double a1_ = 2;
  double a2_ = -1;
  int lambda_ = 15;
  size_t num_stripes_ = 4;
  cv::Rect2f frame_window_;
};

class SIRParticleFiltersNormalHistogram : public SIRParticleFilters {
 public:
  SIRParticleFiltersNormalHistogram(const std::string& config_file,
                                    const Mat3Uchar& frame)
      : SIRParticleFilters(config_file, frame) {}

  void SetupColorSpace(ColorSpaceEnum color_space_enum) override;
  void ComputeNormalizedWeights(const cv::Mat& frame) override;

 private:
  std::unique_ptr<HistogramFactory> histogram_creator_;
};

class SIRParticleFiltersIntegralHistogram : public SIRParticleFilters {
 public:
  SIRParticleFiltersIntegralHistogram(const std::string& config_file,
                                      const Mat3Uchar& frame)
      : SIRParticleFilters(config_file, frame) {}
  void SetupColorSpace(ColorSpaceEnum color_space_enum) override;
  void ComputeNormalizedWeights(const cv::Mat& frame) override;

 private:
  std::unique_ptr<IntegralHistogramFactory> integral_histogram_creator_;
};

}  // namespace jg

#endif  // JG_SIR_PARTICLE_FILTER_H
