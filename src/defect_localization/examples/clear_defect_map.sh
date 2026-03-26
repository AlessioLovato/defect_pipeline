#!/usr/bin/env bash
set -euo pipefail

ros2 service call /defect_map/clear_defect_map defect_map_interfaces/srv/ClearDefectMap "{}"
