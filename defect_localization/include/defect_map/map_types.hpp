/**
 * @file map_types.hpp
 * @brief Shared defect-map data structures used by the in-package map owner node.
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_MAP__MAP_TYPES_HPP_
#define DEFECT_LOCALIZATION__DEFECT_MAP__MAP_TYPES_HPP_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <vector>

#include "defect_map_interfaces/msg/defect_entry.hpp"

namespace defect_localization
{
namespace defect_map
{

template<typename T>
using Expected = std::expected<T, std::string>;

using ExpectedVoid = std::expected<void, std::string>;

/**
 * @brief Shared status payload for internal defect-map operations.
 *
 * The ROS service adapters still expose their own field names, but the core
 * map code now reuses a single success/status/message shape internally.
 */
struct OperationResult
{
  bool success{false};
  std::string status_code;
  std::string message;
};

/**
 * @brief Integer voxel key in the global base-frame voxel grid.
 *
 * The map owner stores geometry as discrete voxel coordinates so entries from
 * different capture zones remain directly comparable.
 */
struct VoxelKey
{
  int32_t x{0};
  int32_t y{0};
  int32_t z{0};

  auto operator<=>(const VoxelKey &) const = default;
};

/**
 * @brief Internal raw defect record owned by the map store.
 */
struct RawDefectRecord
{
  uint64_t uid{0U};
  std::string frame_id;
  std::string zone_id;
  std::string label;
  float score{0.0F};
  std::vector<VoxelKey> voxels;
};

/**
 * @brief Immutable service snapshot published to readers.
 *
 * Readers never access mutable state directly. They consume a prebuilt snapshot
 * that is atomically replaced after successful write/process operations.
 */
struct MapSnapshot
{
  std::vector<defect_map_interfaces::msg::DefectEntry> raw_entries;
  std::vector<defect_map_interfaces::msg::DefectEntry> clustered_entries;
  uint64_t latest_uid{0U};
  uint64_t cluster_epoch{0U};
};

/**
 * @brief Immutable persistence snapshot used for save operations.
 */
struct PersistenceState
{
  std::vector<RawDefectRecord> raw_defects;
  uint64_t latest_uid{0U};
};

/**
 * @brief Small summary of the currently published snapshots.
 */
struct SnapshotMetadata
{
  size_t raw_count{0U};
  size_t clustered_count{0U};
  uint64_t latest_uid{0U};
  uint64_t cluster_epoch{0U};
};

/**
 * @brief Mutable raw-state container owned by the store thread.
 */
using RawDefectMap = std::map<uint64_t, RawDefectRecord>;

}  // namespace defect_map
}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_MAP__MAP_TYPES_HPP_
