/**
 * @file json_persistence.cpp
 * @brief JSON persistence helpers for the defect map node.
 */
#include "defect_map/json_persistence.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace defect_localization
{
namespace defect_map
{
namespace
{

constexpr uint64_t kFormatVersion = 1U;

struct TemporaryFile
{
  int fd{-1};
  std::filesystem::path path;
};

ExpectedVoid writeAll(int fd, std::string_view data)
{
  size_t total_written = 0U;
  while (total_written < data.size()) {
    const auto remaining = data.size() - total_written;
    const auto bytes_written = ::write(fd, data.data() + total_written, remaining);
    if (bytes_written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::unexpected(std::string("write failed: ") + std::strerror(errno));
    }
    if (bytes_written == 0) {
      return std::unexpected(std::string("write returned 0 bytes for a non-empty payload"));
    }
    total_written += static_cast<size_t>(bytes_written);
  }
  return {};
}

Expected<TemporaryFile> createTemporaryFile(const std::filesystem::path & file_path)
{
  auto temp_template = file_path.string() + ".tmp.XXXXXX";
  std::vector<char> path_buffer(temp_template.begin(), temp_template.end());
  path_buffer.push_back('\0');

  const int fd = ::mkstemp(path_buffer.data());
  if (fd < 0) {
    return std::unexpected(std::string("mkstemp failed: ") + std::strerror(errno));
  }

  return TemporaryFile{
    .fd = fd,
    .path = std::filesystem::path(path_buffer.data())};
}

bool fsyncParentDirectory(const std::filesystem::path & file_path)
{
  const auto parent_path = file_path.parent_path();
  if (parent_path.empty()) {
    return true;
  }

  const int dir_fd = ::open(parent_path.c_str(), O_RDONLY | O_DIRECTORY);
  if (dir_fd < 0) {
    return false;
  }

  const bool ok = (::fsync(dir_fd) == 0);
  ::close(dir_fd);
  return ok;
}

template<typename IntT>
Expected<IntT> readInteger(
  const boost::property_tree::ptree & tree,
  const std::string & field_name)
{
  const auto optional_text = tree.get_optional<std::string>(field_name);
  if (!optional_text.has_value()) {
    return std::unexpected(std::string("Missing integer field: ") + field_name);
  }

  try {
    size_t consumed = 0U;
    if constexpr (std::is_signed_v<IntT>) {
      const auto parsed_value = std::stoll(optional_text.value(), &consumed, 10);
      if (
        consumed != optional_text->size() ||
        parsed_value < static_cast<long long>(std::numeric_limits<IntT>::min()) ||
        parsed_value > static_cast<long long>(std::numeric_limits<IntT>::max()))
      {
        return std::unexpected(std::string("Out-of-range integer field: ") + field_name);
      }
      return static_cast<IntT>(parsed_value);
    } else {
      const auto parsed_value = std::stoull(optional_text.value(), &consumed, 10);
      if (
        consumed != optional_text->size() ||
        parsed_value > static_cast<unsigned long long>(std::numeric_limits<IntT>::max()))
      {
        return std::unexpected(std::string("Out-of-range integer field: ") + field_name);
      }
      return static_cast<IntT>(parsed_value);
    }
  } catch (const std::exception &) {
    return std::unexpected(std::string("Invalid integer field: ") + field_name);
  }
}

Expected<VoxelKey> parseVoxelTriple(const boost::property_tree::ptree & triple_tree)
{
  std::vector<int32_t> values;
  values.reserve(3U);

  for (const auto & child : triple_tree) {
    const auto value = readInteger<int32_t>(child.second, "");
    if (!value) {
      return std::unexpected(std::string("Invalid voxel coordinate: ") + value.error());
    }
    values.push_back(*value);
  }

  if (values.size() != 3U) {
    return std::unexpected(std::string("Voxel triple must contain exactly 3 integers"));
  }

  return VoxelKey{values[0], values[1], values[2]};
}

}  // namespace

ExpectedVoid JsonPersistence::save(
  const std::filesystem::path & file_path,
  const PersistenceState & state,
  bool pretty_json) const
{
  try {
    const auto parent_path = file_path.parent_path();
    if (!parent_path.empty()) {
      std::filesystem::create_directories(parent_path);
    }

    boost::property_tree::ptree root;
    root.put("format_version", kFormatVersion);
    root.put("latest_uid", state.latest_uid);

    boost::property_tree::ptree entries_tree;
    for (const auto & raw_record : state.raw_defects) {
      boost::property_tree::ptree entry_tree;
      entry_tree.put("uid", raw_record.uid);
      entry_tree.put("frame_id", raw_record.frame_id);
      entry_tree.put("zone_id", raw_record.zone_id);
      entry_tree.put("label", raw_record.label);
      entry_tree.put("score", raw_record.score);

      boost::property_tree::ptree voxels_tree;
      for (const auto & voxel : raw_record.voxels) {
        boost::property_tree::ptree triple_tree;
        for (const auto coordinate : {voxel.x, voxel.y, voxel.z}) {
          boost::property_tree::ptree coordinate_tree;
          coordinate_tree.put("", coordinate);
          triple_tree.push_back({"", coordinate_tree});
        }
        voxels_tree.push_back({"", triple_tree});
      }
      entry_tree.add_child("voxels", voxels_tree);
      entries_tree.push_back({"", entry_tree});
    }
    root.add_child("entries", entries_tree);

    std::ostringstream buffer;
    boost::property_tree::write_json(buffer, root, pretty_json);
    const auto payload = buffer.str();

    auto temp_file = createTemporaryFile(file_path);
    if (!temp_file) {
      return std::unexpected(temp_file.error());
    }

    int fd = temp_file->fd;
    auto temp_path = std::move(temp_file->path);
    auto cleanupTempFile = [&]() {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
      if (!temp_path.empty()) {
        std::error_code remove_error;
        std::filesystem::remove(temp_path, remove_error);
      }
    };

    // Persist the full payload to a unique temp file before exposing it via rename.
    if (const auto write_result = writeAll(fd, payload); !write_result) {
      cleanupTempFile();
      return std::unexpected(write_result.error());
    }
    if (::fsync(fd) != 0) {
      cleanupTempFile();
      return std::unexpected(std::string("fsync failed: ") + std::strerror(errno));
    }
    if (::close(fd) != 0) {
      fd = -1;
      cleanupTempFile();
      return std::unexpected(std::string("close failed: ") + std::strerror(errno));
    }
    fd = -1;

    try {
      std::filesystem::rename(temp_path, file_path);
      temp_path.clear();
    } catch (...) {
      cleanupTempFile();
      throw;
    }
    (void)fsyncParentDirectory(file_path);
    return {};
  } catch (const std::exception & ex) {
    return std::unexpected(std::string(ex.what()));
  }
}

Expected<PersistenceState> JsonPersistence::load(const std::filesystem::path & file_path) const
{
  try {
    boost::property_tree::ptree root;
    boost::property_tree::read_json(file_path.string(), root);

    const auto format_version = readInteger<uint64_t>(root, "format_version");
    if (!format_version) {
      return std::unexpected(format_version.error());
    }
    if (*format_version != kFormatVersion) {
      return std::unexpected(
        std::string("Unsupported format_version: ") + std::to_string(*format_version));
    }

    const auto latest_uid = readInteger<uint64_t>(root, "latest_uid");
    if (!latest_uid) {
      return std::unexpected(latest_uid.error());
    }

    PersistenceState loaded_state;
    loaded_state.latest_uid = *latest_uid;

    std::set<uint64_t> seen_uids;
    uint64_t max_uid = 0U;

    const auto entries_optional = root.get_child_optional("entries");
    if (entries_optional.has_value()) {
      for (const auto & entry_pair : entries_optional.value()) {
        const auto & entry_tree = entry_pair.second;

        RawDefectRecord raw_record;
        const auto raw_uid = readInteger<uint64_t>(entry_tree, "uid");
        if (!raw_uid) {
          return std::unexpected(raw_uid.error());
        }
        raw_record.uid = *raw_uid;

        if (!seen_uids.insert(raw_record.uid).second) {
          return std::unexpected(
            std::string("Duplicate uid in persisted file: ") + std::to_string(raw_record.uid));
        }

        raw_record.zone_id = entry_tree.get<std::string>("zone_id", "");
        raw_record.frame_id = entry_tree.get<std::string>("frame_id", "");
        raw_record.label = entry_tree.get<std::string>("label", "");
        raw_record.score = entry_tree.get<float>("score", 0.0F);

        const auto voxels_optional = entry_tree.get_child_optional("voxels");
        if (!voxels_optional.has_value()) {
          return std::unexpected(
            std::string("Missing voxels array for uid ") + std::to_string(raw_record.uid));
        }

        std::set<VoxelKey> unique_voxels;
        for (const auto & voxel_pair : voxels_optional.value()) {
          const auto voxel = parseVoxelTriple(voxel_pair.second);
          if (!voxel) {
            return std::unexpected(voxel.error());
          }
          unique_voxels.insert(*voxel);
        }

        if (unique_voxels.empty()) {
          return std::unexpected(
            std::string("Persisted entry has no voxels for uid ") +
            std::to_string(raw_record.uid));
        }

        raw_record.voxels.assign(unique_voxels.begin(), unique_voxels.end());
        loaded_state.raw_defects.push_back(raw_record);
        max_uid = std::max(max_uid, raw_record.uid);
      }
    }

    if (loaded_state.latest_uid < max_uid) {
      return std::unexpected(
        std::string("Persisted latest_uid is smaller than the maximum stored uid: ") +
        std::to_string(loaded_state.latest_uid) + " < " + std::to_string(max_uid));
    }

    return loaded_state;
  } catch (const boost::property_tree::json_parser::json_parser_error & ex) {
    return std::unexpected(std::string(ex.message()));
  } catch (const std::exception & ex) {
    return std::unexpected(std::string(ex.what()));
  }
}

}  // namespace defect_map
}  // namespace defect_localization
