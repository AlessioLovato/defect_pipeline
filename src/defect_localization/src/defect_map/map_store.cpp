/**
 * @file map_store.cpp
 * @brief Implementation of the single-owner defect map store.
 */
#include "defect_map/map_store.hpp"

#include <algorithm>
#include <exception>
#include <set>
#include <sstream>
#include <unordered_set>

namespace defect_localization
{
namespace defect_map
{
namespace
{

constexpr uint32_t kRetryAfterMs = 50U;

defect_map_interfaces::msg::DefectEntry toMessage(
  const RawDefectRecord & raw_record,
  bool clustered)
{
  defect_map_interfaces::msg::DefectEntry entry;
  entry.uid = raw_record.uid;
  entry.cluster = clustered;
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

bool matchesFilters(
  const defect_map_interfaces::msg::DefectEntry & entry,
  const std::string & zone_filter,
  const std::string & label_filter)
{
  if (!zone_filter.empty() && entry.zone_id != zone_filter) {
    return false;
  }
  if (!label_filter.empty() && entry.label != label_filter) {
    return false;
  }
  return true;
}

Expected<RawDefectRecord> normalizeIncomingEntry(
  const defect_map_interfaces::msg::DefectEntry & entry)
{
  if (entry.cluster) {
    return std::unexpected(std::string("Incoming entries must be raw defects (cluster=false)"));
  }
  if (entry.frame_id.empty()) {
    return std::unexpected(std::string("Incoming entries must set frame_id"));
  }

  if (
    entry.voxel_ix.size() != entry.voxel_iy.size() ||
    entry.voxel_ix.size() != entry.voxel_iz.size())
  {
    return std::unexpected(
      std::string("voxel_ix/voxel_iy/voxel_iz arrays must have the same length"));
  }

  if (entry.voxel_ix.empty()) {
    return std::unexpected(std::string("Defect entries must contain at least one voxel"));
  }

  std::set<VoxelKey> unique_voxels;
  for (size_t i = 0; i < entry.voxel_ix.size(); ++i) {
    unique_voxels.insert(VoxelKey{
      entry.voxel_ix[i],
      entry.voxel_iy[i],
      entry.voxel_iz[i]});
  }

  return RawDefectRecord{
    .uid = entry.uid,
    .frame_id = entry.frame_id,
    .zone_id = entry.zone_id,
    .label = entry.label,
    .score = entry.score,
    .voxels = std::vector<VoxelKey>(unique_voxels.begin(), unique_voxels.end())};
}

template<typename ResultT>
ResultT makeBusyResult()
{
  ResultT result;
  result.success = false;
  result.status_code = "BUSY_RETRY";
  result.message = "Map store is busy; retry later";
  if constexpr (requires(ResultT busy_result) { busy_result.retry_after_ms = 0U; }) {
    result.retry_after_ms = kRetryAfterMs;
  }
  return result;
}

template<typename ResultT>
ResultT makeInternalErrorResult(std::string message)
{
  ResultT result;
  result.success = false;
  result.status_code = "INTERNAL_ERROR";
  result.message = std::move(message);
  return result;
}

}  // namespace

MapStore::MapStore()
{
  // Publish a well-defined empty snapshot before any service request arrives.
  rebuildPublishedSnapshots(false);
  owner_thread_ = std::thread(&MapStore::ownerLoop, this);
}

MapStore::~MapStore()
{
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_requested_ = true;
  }
  queue_cv_.notify_all();
  if (owner_thread_.joinable()) {
    owner_thread_.join();
  }
}

AddDefectsResult MapStore::addDefects(
  const std::vector<defect_map_interfaces::msg::DefectEntry> & defects,
  std::chrono::milliseconds wait_timeout)
{
  auto promise = std::make_shared<std::promise<AddDefectsResult>>();
  auto future = promise->get_future();

  enqueueWrite([this, defects, promise]() {
    try {
      promise->set_value(doAddDefects(defects));
    } catch (const std::exception & ex) {
      auto result = makeInternalErrorResult<AddDefectsResult>(ex.what());
      result.rejected_count = static_cast<uint32_t>(defects.size());
      promise->set_value(std::move(result));
    }
  });

  return waitForFuture(future, wait_timeout, makeBusyResult<AddDefectsResult>());
}

RemoveDefectsResult MapStore::removeDefects(
  const std::vector<uint64_t> & uids,
  bool clustered_uids,
  std::chrono::milliseconds wait_timeout)
{
  auto promise = std::make_shared<std::promise<RemoveDefectsResult>>();
  auto future = promise->get_future();

  enqueueWrite([this, uids, clustered_uids, promise]() {
    try {
      promise->set_value(doRemoveDefects(uids, clustered_uids));
    } catch (const std::exception & ex) {
      promise->set_value(makeInternalErrorResult<RemoveDefectsResult>(ex.what()));
    }
  });

  return waitForFuture(future, wait_timeout, makeBusyResult<RemoveDefectsResult>());
}

GetDefectsResult MapStore::getDefects(
  bool clustered_view,
  const std::string & zone_filter,
  const std::string & label_filter) const
{
  std::shared_ptr<const MapSnapshot> snapshot;
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot = published_snapshot_;
  }

  GetDefectsResult result;
  result.cluster_epoch = snapshot ? snapshot->cluster_epoch : 0U;

  if (!snapshot) {
    result.success = false;
    result.status_code = "INTERNAL_ERROR";
    result.message = "No published snapshot is available";
    return result;
  }

  const auto & source_entries =
    clustered_view ? snapshot->clustered_entries : snapshot->raw_entries;
  result.entries.reserve(source_entries.size());

  for (const auto & entry : source_entries) {
    if (matchesFilters(entry, zone_filter, label_filter)) {
      result.entries.push_back(entry);
    }
  }

  if (result.entries.empty()) {
    result.success = false;
    result.status_code = "NO_DATA";
    result.message = "No defect entries match the requested filters";
    return result;
  }

  result.success = true;
  result.status_code = "OK";
  result.message = "Defect entries retrieved";
  return result;
}

ProcessClustersResult MapStore::processClusters(
  bool force_recompute,
  std::chrono::milliseconds wait_timeout)
{
  auto promise = std::make_shared<std::promise<ProcessClustersResult>>();
  auto future = promise->get_future();

  enqueueProcess([this, force_recompute, promise]() {
    try {
      promise->set_value(doProcessClusters(force_recompute));
    } catch (const std::exception & ex) {
      promise->set_value(makeInternalErrorResult<ProcessClustersResult>(ex.what()));
    }
  });

  return waitForFuture(future, wait_timeout, makeBusyResult<ProcessClustersResult>());
}

ReplaceStateResult MapStore::replaceState(
  const PersistenceState & state,
  std::chrono::milliseconds wait_timeout)
{
  auto promise = std::make_shared<std::promise<ReplaceStateResult>>();
  auto future = promise->get_future();

  enqueueWrite([this, state, promise]() {
    try {
      promise->set_value(doReplaceState(state));
    } catch (const std::exception & ex) {
      promise->set_value(makeInternalErrorResult<ReplaceStateResult>(ex.what()));
    }
  });

  return waitForFuture(future, wait_timeout, makeBusyResult<ReplaceStateResult>());
}

ClearMapResult MapStore::clear(std::chrono::milliseconds wait_timeout)
{
  auto promise = std::make_shared<std::promise<ClearMapResult>>();
  auto future = promise->get_future();

  enqueueWrite([this, promise]() {
    try {
      promise->set_value(doClear());
    } catch (const std::exception & ex) {
      promise->set_value(makeInternalErrorResult<ClearMapResult>(ex.what()));
    }
  });

  return waitForFuture(future, wait_timeout, makeBusyResult<ClearMapResult>());
}

PersistenceState MapStore::capturePersistenceState() const
{
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  return published_persistence_state_ ? *published_persistence_state_ : PersistenceState{};
}

SnapshotMetadata MapStore::snapshotMetadata() const
{
  std::lock_guard<std::mutex> lock(snapshot_mutex_);

  SnapshotMetadata metadata;
  if (published_snapshot_) {
    metadata.raw_count = published_snapshot_->raw_entries.size();
    metadata.clustered_count = published_snapshot_->clustered_entries.size();
    metadata.latest_uid = published_snapshot_->latest_uid;
    metadata.cluster_epoch = published_snapshot_->cluster_epoch;
  }
  return metadata;
}

void MapStore::ownerLoop()
{
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() {
        return stop_requested_ || !write_queue_.empty() || !process_queue_.empty();
      });

      if (stop_requested_ && write_queue_.empty() && process_queue_.empty()) {
        return;
      }

      // Writers have priority over processing jobs.
      if (!write_queue_.empty()) {
        task = std::move(write_queue_.front());
        write_queue_.pop_front();
      } else if (!process_queue_.empty()) {
        task = std::move(process_queue_.front());
        process_queue_.pop_front();
      }
    }

    if (task) {
      task();
    }
  }
}

void MapStore::rebuildPublishedSnapshots(bool increment_cluster_epoch)
{
  if (increment_cluster_epoch) {
    ++cluster_epoch_;
  }

  auto snapshot = std::make_shared<MapSnapshot>();
  auto persistence_state = std::make_shared<PersistenceState>();
  snapshot->latest_uid = latest_uid_;
  snapshot->cluster_epoch = cluster_epoch_;
  persistence_state->latest_uid = latest_uid_;

  snapshot->raw_entries.reserve(raw_defects_.size());
  persistence_state->raw_defects.reserve(raw_defects_.size());

  for (const auto & [uid, raw_record] : raw_defects_) {
    (void)uid;
    snapshot->raw_entries.push_back(toMessage(raw_record, false));
    persistence_state->raw_defects.push_back(raw_record);
  }

  snapshot->clustered_entries = clusterer_.buildClusteredEntries(raw_defects_);

  // Replace both snapshots atomically from the readers' perspective.
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  published_snapshot_ = std::move(snapshot);
  published_persistence_state_ = std::move(persistence_state);
}

AddDefectsResult MapStore::doAddDefects(
  const std::vector<defect_map_interfaces::msg::DefectEntry> & defects)
{
  AddDefectsResult result;
  result.latest_uid = latest_uid_;

  if (defects.empty()) {
    result.success = false;
    result.status_code = "INVALID_INPUT";
    result.message = "At least one defect entry is required";
    return result;
  }

  std::vector<RawDefectRecord> normalized_records;
  normalized_records.reserve(defects.size());

  uint64_t expected_uid = latest_uid_ + 1U;
  for (const auto & entry : defects) {
    if (entry.uid != expected_uid || raw_defects_.count(entry.uid) > 0U) {
      result.success = false;
      result.status_code = "UID_OUT_OF_SYNC";
      result.message =
        "Incoming UID sequence does not match the current map counter";
      result.latest_uid = latest_uid_;
      result.rejected_count = static_cast<uint32_t>(defects.size());
      return result;
    }

    const auto raw_record = normalizeIncomingEntry(entry);
    if (!raw_record) {
      result.success = false;
      result.status_code = "INVALID_INPUT";
      result.message = raw_record.error();
      result.latest_uid = latest_uid_;
      result.rejected_count = static_cast<uint32_t>(defects.size());
      return result;
    }

    normalized_records.push_back(*raw_record);
    ++expected_uid;
  }

  for (const auto & raw_record : normalized_records) {
    raw_defects_.emplace(raw_record.uid, raw_record);
  }

  latest_uid_ = expected_uid - 1U;
  rebuildPublishedSnapshots(true);

  result.success = true;
  result.status_code = "OK";
  result.message = "Defect batch accepted";
  result.latest_uid = latest_uid_;
  result.accepted_count = static_cast<uint32_t>(normalized_records.size());
  result.rejected_count = 0U;
  result.retry_after_ms = 0U;
  return result;
}

RemoveDefectsResult MapStore::doRemoveDefects(
  const std::vector<uint64_t> & uids,
  bool clustered_uids)
{
  (void)clustered_uids;

  RemoveDefectsResult result;
  if (uids.empty()) {
    result.success = false;
    result.status_code = "NOT_FOUND";
    result.message = "No UIDs were provided";
    return result;
  }

  std::set<uint64_t> unique_uids(uids.begin(), uids.end());
  for (const auto uid : unique_uids) {
    const auto erased = raw_defects_.erase(uid);
    if (erased > 0U) {
      ++result.removed_count;
    } else {
      result.not_found_uids.push_back(uid);
    }
  }

  if (result.removed_count == 0U) {
    result.success = false;
    result.status_code = "NOT_FOUND";
    result.message = "Requested UIDs are not present in the map";
    return result;
  }

  rebuildPublishedSnapshots(true);

  result.success = true;
  if (result.not_found_uids.empty()) {
    result.status_code = "OK";
    result.message = "Requested defects removed";
  } else {
    result.status_code = "PARTIAL";
    result.message = "Some requested defects were removed";
  }
  return result;
}

ProcessClustersResult MapStore::doProcessClusters(bool force_recompute)
{
  (void)force_recompute;

  // Phase-1 always rebuilds from the raw source of truth.
  rebuildPublishedSnapshots(true);

  ProcessClustersResult result;
  result.success = true;
  result.status_code = "OK";
  result.message = "Clustered view refreshed";
  result.cluster_epoch = cluster_epoch_;
  result.cluster_count = static_cast<uint32_t>(raw_defects_.size());
  return result;
}

ReplaceStateResult MapStore::doReplaceState(const PersistenceState & state)
{
  ReplaceStateResult result;

  raw_defects_.clear();
  uint64_t max_uid = 0U;
  for (const auto & raw_record : state.raw_defects) {
    raw_defects_[raw_record.uid] = raw_record;
    max_uid = std::max(max_uid, raw_record.uid);
  }

  latest_uid_ = std::max(state.latest_uid, max_uid);
  cluster_epoch_ = 0U;
  rebuildPublishedSnapshots(true);

  result.success = true;
  result.status_code = "OK";
  result.message = "Map state loaded";
  result.loaded_entries = static_cast<uint32_t>(raw_defects_.size());
  result.latest_uid = latest_uid_;
  return result;
}

ClearMapResult MapStore::doClear()
{
  ClearMapResult result;

  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    if (published_snapshot_) {
      result.cleared_latest_raw_entries =
        static_cast<uint32_t>(published_snapshot_->raw_entries.size());
      result.cleared_latest_clustered_entries =
        static_cast<uint32_t>(published_snapshot_->clustered_entries.size());
    }
  }

  result.cleared_raw_entries = static_cast<uint32_t>(raw_defects_.size());

  raw_defects_.clear();
  latest_uid_ = 0U;
  cluster_epoch_ = 0U;
  rebuildPublishedSnapshots(false);

  result.success = true;
  result.status_code = "OK";
  result.message = "Defect map cleared";
  result.cleared_pending_images = 0U;
  result.cleared_queued_jobs = 0U;
  return result;
}

void MapStore::enqueueWrite(std::function<void()> task)
{
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    write_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_one();
}

void MapStore::enqueueProcess(std::function<void()> task)
{
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    process_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_one();
}

}  // namespace defect_map
}  // namespace defect_localization
