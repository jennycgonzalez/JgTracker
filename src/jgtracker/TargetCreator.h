#ifndef JG_TARGET_CREATOR_H
#define JG_TARGET_CREATOR_H

#include "jgtracker/types.h"
#include "jgtracker/Target.h"

namespace jg {

void AddNewTarget(std::vector<Target>& targets, const cv::Rect2f& box,
                  const Mat3Uchar& frame);

void AddTargetFromFile(std::vector<Target>& targets,
                       const boost::filesystem::path& filename,
                       const Mat3Uchar& frame);

}  // namespace jg

#endif  // JG_TARGET_CREATOR_H
