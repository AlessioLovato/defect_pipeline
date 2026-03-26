/**
 * @file json_persistence.hpp
 * @brief JSON save/load helper for the defect map node.
 */
#ifndef DEFECT_LOCALIZATION__DEFECT_MAP__JSON_PERSISTENCE_HPP_
#define DEFECT_LOCALIZATION__DEFECT_MAP__JSON_PERSISTENCE_HPP_

#include <filesystem>
#include <string>

#include "defect_map/map_types.hpp"

namespace defect_localization
{
namespace defect_map
{

/**
 * @brief Handles JSON persistence of the authoritative raw defect state.
 *
 * Clustered snapshots are intentionally not serialized because they are a
 * derived view that can be rebuilt from the raw records after load.
 */
class JsonPersistence
{
public:
  /**
   * @brief Save a raw map snapshot to disk using atomic replace semantics.
   * @param file_path Destination path.
   * @param state Raw persistence snapshot to save.
   * @param pretty_json When true, emit indented JSON.
   * @return Empty success or a detailed error string on failure.
   */
  ExpectedVoid save(
    const std::filesystem::path & file_path,
    const PersistenceState & state,
    bool pretty_json) const;

  /**
   * @brief Load a raw map snapshot from a JSON file.
   * @param file_path Source path.
   * @return Parsed raw persistence snapshot or a detailed error string.
   */
  Expected<PersistenceState> load(const std::filesystem::path & file_path) const;
};

}  // namespace defect_map
}  // namespace defect_localization

#endif  // DEFECT_LOCALIZATION__DEFECT_MAP__JSON_PERSISTENCE_HPP_
