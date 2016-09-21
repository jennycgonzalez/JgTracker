#ifndef JG_MANAGER_H
#define JG_MANAGER_H

#include <set>
#include <opencv2/core.hpp>
#include "jg/types.h"
#include "jg/Target.h"
#include "jg/TargetCreator.h"
#include "jg/TargetSelector.h"

namespace jg {

//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

class Display {
 public:
  Display(const std::string& config_file);
  virtual ~Display() = default;
  virtual void ShowPotentialTargets(const std::vector<Target>& targets,
                                    Mat3Uchar& frame,
                                    const std::string& window_name,
                                    const cv::Scalar& color) const = 0;
  virtual void ShowTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                           const std::string& window_name) const = 0;
  virtual void ShowAllTargets(const std::vector<Target>& targets,
                              Mat3Uchar& frame,
                              const std::string& window_name) const = 0;

 protected:
  std::size_t min_visible_count_ = 5;
  std::size_t max_continuous_invisible_count_ = 2;
};

class InactiveDisplay : public Display {
 public:
  InactiveDisplay(const std::string& config_file) : Display(config_file) {}

  void ShowPotentialTargets(const std::vector<Target>& targets,
                            Mat3Uchar& frame, const std::string& window_name,
                            const cv::Scalar& color) const override;
  void ShowTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                   const std::string& window_name) const override;
  void ShowAllTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                      const std::string& window_name) const override;
};

class ActiveDisplay : public Display {
 public:
  ActiveDisplay(const std::string& config_file) : Display(config_file) {}

  void ShowPotentialTargets(const std::vector<Target>& targets,
                            Mat3Uchar& frame, const std::string& window_name,
                            const cv::Scalar& color) const override;
  void ShowTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                   const std::string& window_name) const override;
  void ShowAllTargets(const std::vector<Target>& targets, Mat3Uchar& frame,
                      const std::string& window_name) const override;
};

//------------------------------------------------------------------------------
// Manager
//------------------------------------------------------------------------------

class Manager {
 public:
  Manager() = default;
  Manager(Manager&&) = default;
  Manager& operator=(Manager&&) = default;
  Manager(int delay,           const std::string& config_file);

  void DeleteLostTargets(std::vector<Target>& targets) const;

  const Display* display() const { return display_.get(); }

 private:
  double next_id_ = 0;
  bool show_intermediate_results_;
  std::size_t max_continuous_invisible_count_ = 10;
  std::size_t age_threshold_ = 8;
  double max_invisibility_ratio_ = 0.8;
  std::unique_ptr<Display> display_;
};

}  // namespace jg

#endif  // JG_MANAGER_H
