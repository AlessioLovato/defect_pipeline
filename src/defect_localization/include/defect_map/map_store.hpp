/**
 * @file map_store.hpp
 * @brief Single-owner defect-map state store with immutable reader snapshots.
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_MAP__MAP_STORE_HPP_
#define DEFECT_LOCALIZATION__DEFECT_MAP__MAP_STORE_HPP_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "defect_map_interfaces/msg/defect_entry.hpp"
#include "defect_map/clusterer_mock.hpp"
#include "defect_map/map_types.hpp"

namespace defect_localization
{
namespace defect_map
{

/**
 * @brief Result returned by AddDefects store operations.
 */
struct AddDefectsResult
  : OperationResult
{
  uint64_t latest_uid{0U};
  uint32_t accepted_count{0U};
  uint32_t rejected_count{0U};
  uint32_t retry_after_ms{0U};
};

/**
 * @brief Result returned by RemoveDefects store operations.
 */
struct RemoveDefectsResult
  : OperationResult
{
  uint32_t removed_count{0U};
  std::vector<uint64_t> not_found_uids;
};

/**
 * @brief Result returned by query operations.
 */
struct GetDefectsResult
  : OperationResult
{
  std::vector<defect_map_interfaces::msg::DefectEntry> entries;
  uint64_t cluster_epoch{0U};
};

/**
 * @brief Result returned by clustered-view processing operations.
 */
struct ProcessClustersResult
  : OperationResult
{
  uint64_t cluster_epoch{0U};
  uint32_t cluster_count{0U};
};

/**
 * @brief Result returned when replacing state from a loaded snapshot.
 */
struct ReplaceStateResult
  : OperationResult
{
  uint32_t loaded_entries{0U};
  uint64_t latest_uid{0U};
};

/**
 * @brief Result returned by clear operations.
 */
struct ClearMapResult
  : OperationResult
{
  uint32_t cleared_raw_entries{0U};
  uint32_t cleared_latest_raw_entries{0U};
  uint32_t cleared_latest_clustered_entries{0U};
  uint32_t cleared_pending_images{0U};
  uint32_t cleared_queued_jobs{0U};
};

/**
 * @brief Authoritative raw-state owner for the defect map node.
 *
 * Writes and processing are serialized through a dedicated owner thread. Read
 * paths consume immutable snapshots and never touch the mutable raw map.
 */
class MapStore
{
public:
  /**
   * @brief Start the owner thread and publish an empty initial snapshot.
   */
  MapStore();

  /**
   * @brief Stop the owner thread and discard queued work.
   */
  ~MapStore();

  /**
   * @brief Validate and append a contiguous batch of raw defects.
   * @param defects Incoming defect entries from the pipeline.
   * @param wait_timeout How long the caller will wait for the owner thread.
   * @return Operation result including UID resync information on conflicts.
   */
  AddDefectsResult addDefects(
    const std::vector<defect_map_interfaces::msg::DefectEntry> & defects,
    std::chrono::milliseconds wait_timeout);

  /**
   * @brief Remove defects by UID from the authoritative raw state.
   * @param uids Requested raw or clustered UIDs.
   * @param clustered_uids When true, interpret the UIDs as clustered-view IDs.
   * @param wait_timeout How long the caller will wait for the owner thread.
   * @return Removal status with partial/not-found details.
   */
  RemoveDefectsResult removeDefects(
    const std::vector<uint64_t> & uids,
    bool clustered_uids,
    std::chrono::milliseconds wait_timeout);

  /**
   * @brief Read a filtered raw or clustered snapshot.
   * @param clustered_view Select the clustered snapshot instead of the raw one.
   * @param zone_filter Optional zone filter; empty means all zones.
   * @param label_filter Optional label filter; empty means all labels.
   * @return Snapshot query result with deterministic ordering.
   */
  GetDefectsResult getDefects(
    bool clustered_view,
    const std::string & zone_filter,
    const std::string & label_filter) const;

  /**
   * @brief Force or refresh the clustered view from the current raw state.
   * @param force_recompute Kept for interface parity; phase-1 always recomputes.
   * @param wait_timeout How long the caller will wait for the owner thread.
   * @return Cluster processing result including the new epoch value.
   */
  ProcessClustersResult processClusters(
    bool force_recompute,
    std::chrono::milliseconds wait_timeout);

  /**
   * @brief Replace the full raw state from a previously loaded snapshot.
   * @param state Loaded persistence state.
   * @param wait_timeout How long the caller will wait for the owner thread.
   * @return Replacement result with loaded entry count and latest UID.
   */
  ReplaceStateResult replaceState(
    const PersistenceState & state,
    std::chrono::milliseconds wait_timeout);

  /**
   * @brief Clear the map and reset counters.
   * @param wait_timeout How long the caller will wait for the owner thread.
   * @return Counts describing the cleared snapshots.
   */
  ClearMapResult clear(std::chrono::milliseconds wait_timeout);

  /**
   * @brief Capture the raw persistence snapshot currently visible to readers.
   * @return Immutable copy of the latest persistence snapshot.
   */
  PersistenceState capturePersistenceState() const;

  /**
   * @brief Read current raw/clustered snapshot sizes for logging.
   * @return Summary of the currently published snapshots.
   */
  SnapshotMetadata snapshotMetadata() const;

private:
  /**
   * @brief Run queued write/process operations with writer-first priority.
   */
  void ownerLoop();

  /**
   * @brief Rebuild the published snapshots from the current raw state.
   * @param increment_cluster_epoch When true, bump the epoch before publishing.
   */
  void rebuildPublishedSnapshots(bool increment_cluster_epoch);

  /**
   * @brief Internal add operation executed on the owner thread.
   * @param defects Incoming raw defects.
   * @return Validated add result.
   */
  AddDefectsResult doAddDefects(
    const std::vector<defect_map_interfaces::msg::DefectEntry> & defects);

  /**
   * @brief Internal remove operation executed on the owner thread.
   * @param uids Requested UIDs.
   * @param clustered_uids Whether the input refers to clustered IDs.
   * @return Removal result.
   */
  RemoveDefectsResult doRemoveDefects(const std::vector<uint64_t> & uids, bool clustered_uids);

  /**
   * @brief Internal clustered-view refresh executed on the owner thread.
   * @param force_recompute Interface parity flag.
   * @return Process result with the new epoch value.
   */
  ProcessClustersResult doProcessClusters(bool force_recompute);

  /**
   * @brief Internal state replacement executed on the owner thread.
   * @param state Loaded raw state.
   * @return Replacement result.
   */
  ReplaceStateResult doReplaceState(const PersistenceState & state);

  /**
   * @brief Internal clear executed on the owner thread.
   * @return Clear result with pre-clear snapshot counts.
   */
  ClearMapResult doClear();

  /**
   * @brief Push work into the writer-priority queue.
   * @param task Task closure to run on the owner thread.
   */
  void enqueueWrite(std::function<void()> task);

  /**
   * @brief Push work into the lower-priority processing queue.
   * @param task Task closure to run on the owner thread.
   */
  void enqueueProcess(std::function<void()> task);

  template<typename ResultT>
  ResultT waitForFuture(
    std::future<ResultT> & future,
    std::chrono::milliseconds wait_timeout,
    const ResultT & busy_result) const
  {
    if (future.wait_for(wait_timeout) != std::future_status::ready) {
      return busy_result;
    }
    return future.get();
  }

  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<std::function<void()>> write_queue_;
  std::deque<std::function<void()>> process_queue_;
  bool stop_requested_{false};
  std::thread owner_thread_;

  mutable std::mutex snapshot_mutex_;
  std::shared_ptr<const MapSnapshot> published_snapshot_;
  std::shared_ptr<const PersistenceState> published_persistence_state_;

  RawDefectMap raw_defects_;
  uint64_t latest_uid_{0U};
  uint64_t cluster_epoch_{0U};
  ClustererMock clusterer_;
};

}  // namespace defect_map
}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_MAP__MAP_STORE_HPP_
