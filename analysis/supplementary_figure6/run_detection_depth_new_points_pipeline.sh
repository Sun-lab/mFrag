#!/usr/bin/env bash
# Build 0.1/1/5/10% detection-depth matrices, run inference + bootstrap, regenerate Figure S6 plots.
set -euo pipefail
CFCRAFT="/fh/working/sun_w/saishi_cui/cfCraft"
PY="${CFCRAFT}/Fragle/software/Fragle_env/bin/python"
DEPTH_ROOT="${CFCRAFT}/Mathios/Final_data_24/detection depth"
BUILD="${CFCRAFT}/Mathios/scripts/build_final_data_24_detection_depth_testing.py"
INFER="${CFCRAFT}/mFrag_Figures/Figure6/plot_figure6_detection_depth_bootstrap_boxplot.py"
PLOT="${CFCRAFT}/mFrag_Figures/Figure S6/plot_figure_s6_detection_depth_bootstrap_trend_boxplot.py"
NEW_DEPTHS="0.1%,1%,5%,10%"
N_REQUIRED=431
LOG="${CFCRAFT}/mFrag_Figures/Figure S6/detection_depth_new_points_pipeline.log"

exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] Waiting for ${NEW_DEPTHS} build (${N_REQUIRED} samples each)..."
while true; do
  ok=1
  for d in 0.1% 1% 5% 10%; do
    n=$(find "${DEPTH_ROOT}/${d}/testing_fragment_length_zscored" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
    echo "  ${d}: ${n}/${N_REQUIRED}"
    if [[ "${n}" -lt "${N_REQUIRED}" ]]; then
      ok=0
    fi
  done
  if [[ "${ok}" -eq 1 ]]; then
    break
  fi
  sleep 300
done

echo "[$(date -Is)] Running inference + bootstrap for ${NEW_DEPTHS}..."
"${PY}" "${INFER}" --inference-only --depths "${NEW_DEPTHS}" --n-required "${N_REQUIRED}"

echo "[$(date -Is)] Regenerating Figure S6 plots (9 depths)..."
"${PY}" "${PLOT}"

echo "[$(date -Is)] Done."
