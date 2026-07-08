#!/usr/bin/env bash
set -euo pipefail

# Calibrated GroundingSAM2 + UniDepthV2 cache for SUNRGBD.
#
# The previous 0.20/0.20 front-end was high-recall but injected too many
# low-quality masks into the 3D PCA stage. This wrapper keeps the upgraded
# models, but makes the 2D proposal distribution closer to the original
# GroundingSAM cache before any 3D fitting happens.

OUTPUT_ROOT="${OUTPUT_ROOT:-pseudo_label_gsam2_udv2_calibrated}"
BASE_GROUND_ROOT="${BASE_GROUND_ROOT:-pseudo_label}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.25}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.20}"
USE_LARGE_GDINO="${USE_LARGE_GDINO:-1}"
UNIDEPTH_LOCAL_FILES_ONLY="${UNIDEPTH_LOCAL_FILES_ONLY:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT}" \
BASE_GROUND_ROOT="${BASE_GROUND_ROOT}" \
BOX_THRESHOLD="${BOX_THRESHOLD}" \
TEXT_THRESHOLD="${TEXT_THRESHOLD}" \
USE_LARGE_GDINO="${USE_LARGE_GDINO}" \
UNIDEPTH_LOCAL_FILES_ONLY="${UNIDEPTH_LOCAL_FILES_ONLY}" \
bash scripts/build_gsam2_unidepthv2_cache_sun.sh "$@"
