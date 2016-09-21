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
