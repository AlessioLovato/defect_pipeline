#!/usr/bin/env bash
set -euo pipefail

NODE_NAME="${1:-/wall_patch_planner}"
ROOM_ID="${2:-}"
WALL_IDS_CSV="${3:-}"

NODE_NAMESPACE="$(dirname "${NODE_NAME}")"
if [ "${NODE_NAMESPACE}" = "." ] || [ "${NODE_NAMESPACE}" = "/" ]; then
  SERVICE_NAME="/plan_patches"
else
  SERVICE_NAME="${NODE_NAMESPACE}/plan_patches"
fi

if [ -n "${ROOM_ID}" ]; then
  ros2 param set "${NODE_NAME}" selected_room_id "${ROOM_ID}"
fi

if [ -n "${WALL_IDS_CSV}" ]; then
  IFS=',' read -r -a WALL_ARRAY <<< "${WALL_IDS_CSV}"
  WALL_PARAM="["
  for i in "${!WALL_ARRAY[@]}"; do
    if [ "$i" -gt 0 ]; then
      WALL_PARAM+=", "
    fi
    WALL_PARAM+="${WALL_ARRAY[$i]}"
  done
  WALL_PARAM+="]"
  ros2 param set "${NODE_NAME}" selected_wall_ids "${WALL_PARAM}"
fi

ros2 service call "${SERVICE_NAME}" std_srvs/srv/Trigger "{}"
