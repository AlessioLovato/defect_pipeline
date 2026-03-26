/**
 * @file mock_image_catalog.hpp
 * @brief Shared mock-image catalog helpers for snapshot testing.
 */
#ifndef DEFECT_LOCALIZATION__MOCK_IMAGE_CATALOG_HPP_
#define DEFECT_LOCALIZATION__MOCK_IMAGE_CATALOG_HPP_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace defect_localization
{

/**
 * @brief One concrete shot file inside a mock image-set folder.
 */
struct MockShotSpec
{
  uint32_t shot_id{0U};
  std::filesystem::path file_path;
};

/**
 * @brief One logical image_id with its full ordered set of mock shots.
 */
struct MockImageSet
{
  std::string image_id;
  std::vector<MockShotSpec> shots;
};

using MockCatalog = std::vector<MockImageSet>;

/**
 * @brief Resolve the installed default root that contains mock_images.
 * @return Installed package-share path for the mock image catalog.
 */
std::filesystem::path defaultMockImagesRoot();

/**
 * @brief Parse and validate the mock_images folder structure.
 * @param images_root Root directory containing one folder per image_id.
 * @param expected_shots_per_image Required number of shots in each folder.
 * @return Ordered mock catalog or an explanatory validation error.
 */
std::expected<MockCatalog, std::string> loadMockCatalog(
  const std::filesystem::path & images_root,
  int expected_shots_per_image);

}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__MOCK_IMAGE_CATALOG_HPP_
