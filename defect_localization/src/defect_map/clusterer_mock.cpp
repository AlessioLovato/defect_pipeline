/**
 * @file clusterer_mock.cpp
 * @brief Implementation of the phase-1 mock clusterer.
 */
#include "defect_map/clusterer_mock.hpp"

namespace defect_localization
{
namespace defect_map
{
namespace
{

defect_map_interfaces::msg::DefectEntry toClusterEntry(const RawDefectRecord & raw_record)
{
  defect_map_interfaces::msg::DefectEntry entry;
  entry.uid = raw_record.uid;
  entry.cluster = true;
  entry.frame_id = raw_record.frame_id;
  entry.zone_id = raw_record.zone_id;
  entry.label = raw_record.label;
  entry.score = raw_record.score;
  entry.voxel_ix.reserve(raw_record.voxels.size());
  entry.voxel_iy.reserve(raw_record.voxels.size());
  entry.voxel_iz.reserve(raw_record.voxels.size());
  for (const auto & voxel : raw_record.voxels) {
    entry.voxel_ix.push_back(voxel.x);
    entry.voxel_iy.push_back(voxel.y);
    entry.voxel_iz.push_back(voxel.z);
  }
  return entry;
}

}  // namespace

std::vector<defect_map_interfaces::msg::DefectEntry> ClustererMock::buildClusteredEntries(
  const RawDefectMap & raw_defects) const
{
  std::vector<defect_map_interfaces::msg::DefectEntry> clustered_entries;
  clustered_entries.reserve(raw_defects.size());

  // Phase-1 clustering is intentionally a 1:1 mirrored view.
  for (const auto & [uid, raw_record] : raw_defects) {
    (void)uid;
    clustered_entries.push_back(toClusterEntry(raw_record));
  }

  return clustered_entries;
}

}  // namespace defect_map
}  // namespace defect_localization
