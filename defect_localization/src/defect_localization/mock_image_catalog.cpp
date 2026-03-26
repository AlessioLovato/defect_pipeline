/**
 * @file mock_image_catalog.cpp
 * @brief Shared mock-image catalog helpers for snapshot testing.
 */
#include "defect_localization/mock_image_catalog.hpp"

#include <algorithm>
#include <regex>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace defect_localization
{
namespace
{

const std::regex kShotPattern(R"((.+)_inst_(\d+)\.png)");

}  // namespace

std::filesystem::path defaultMockImagesRoot()
{
  return
    std::filesystem::path(ament_index_cpp::get_package_share_directory("defect_localization")) /
    "test" / "mock_images";
}

std::expected<MockCatalog, std::string> loadMockCatalog(
  const std::filesystem::path & images_root,
  int expected_shots_per_image)
{
  if (expected_shots_per_image <= 0) {
    return std::unexpected("expected_shots_per_image must be positive");
  }

  if (!std::filesystem::exists(images_root)) {
    return std::unexpected("Mock image root does not exist: " + images_root.string());
  }

  MockCatalog catalog;
  std::vector<std::filesystem::path> folders;
  for (const auto & entry : std::filesystem::directory_iterator(images_root)) {
    if (entry.is_directory()) {
      folders.push_back(entry.path());
    }
  }
  std::sort(folders.begin(), folders.end());

  for (const auto & folder : folders) {
    std::vector<MockShotSpec> shots;

    for (const auto & entry : std::filesystem::directory_iterator(folder)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      std::smatch match;
      const auto file_name = entry.path().filename().string();
      if (!std::regex_match(file_name, match, kShotPattern)) {
        continue;
      }

      const auto image_id = match[1].str();
      if (image_id != folder.filename().string()) {
        return std::unexpected(
          "Mock image folder '" + folder.filename().string() +
          "' contains file with mismatched image_id: " + file_name);
      }

      const auto shot_id = static_cast<uint32_t>(std::stoul(match[2].str()));
      shots.push_back(MockShotSpec{
        .shot_id = shot_id,
        .file_path = entry.path()});
    }

    if (shots.empty()) {
      continue;
    }

    std::sort(
      shots.begin(),
      shots.end(),
      [](const MockShotSpec & lhs, const MockShotSpec & rhs) { return lhs.shot_id < rhs.shot_id; });

    if (static_cast<int>(shots.size()) != expected_shots_per_image) {
      return std::unexpected(
        "Mock image folder '" + folder.filename().string() + "' has " +
        std::to_string(shots.size()) + " shots, expected " +
        std::to_string(expected_shots_per_image));
    }

    for (int i = 0; i < expected_shots_per_image; ++i) {
      const auto expected_shot_id = static_cast<uint32_t>(i + 1);
      if (shots[static_cast<size_t>(i)].shot_id != expected_shot_id) {
        return std::unexpected(
          "Mock image folder '" + folder.filename().string() +
          "' must contain contiguous shot ids from 1 to " +
          std::to_string(expected_shots_per_image));
      }
    }

    catalog.push_back(MockImageSet{
      .image_id = folder.filename().string(),
      .shots = std::move(shots)});
  }

  if (catalog.empty()) {
    return std::unexpected("No valid mock image sets found in " + images_root.string());
  }

  return catalog;
}

}  // namespace defect_localization
