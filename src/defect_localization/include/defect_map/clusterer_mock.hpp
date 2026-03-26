/**
 * @file clusterer_mock.hpp
 * @brief Phase-1 clustered-view generator for the defect map node.
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_MAP__CLUSTERER_MOCK_HPP_
#define DEFECT_LOCALIZATION__DEFECT_MAP__CLUSTERER_MOCK_HPP_

#include <vector>

#include "defect_map/map_types.hpp"

namespace defect_localization
{
namespace defect_map
{

/**
 * @brief Generates the temporary phase-1 clustered view.
 *
 * The approved phase-1 behavior is intentionally simple: every raw defect is
 * mirrored into exactly one clustered entry and marked with `cluster=true`.
 */
class ClustererMock
{
public:
  /**
   * @brief Build the clustered view from the authoritative raw map.
   * @param raw_defects Raw defects indexed by UID.
   * @return Clustered entries in deterministic UID order.
   */
  std::vector<defect_map_interfaces::msg::DefectEntry> buildClusteredEntries(
    const RawDefectMap & raw_defects) const;
};

}  // namespace defect_map
}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_MAP__CLUSTERER_MOCK_HPP_
