#!/usr/bin/env bash
# Wait for detection-depth CSV build (431 testing samples × 4 depths), then run Figure 6 inference + plots.
set -euo pipefail
CFCRAFT="/fh/working/sun_w/saishi_cui/cfCraft"
PY="${CFCRAFT}/Fragle/software/Fragle_env/bin/python"
DEPTH_ROOT="${CFCRAFT}/Mathios/Final_data_24/detection depth"
SCRIPT="${CFCRAFT}/mFrag_Figures/Figure6/plot_figure6_detection_depth_bootstrap_boxplot.py"
N_REQUIRED=431

echo "[wait] Checking for ${N_REQUIRED} samples in each depth folder..."
while true; do
  ok=1
  for d in 20% 40% 60% 80%; do
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

echo "[run] All depth folders complete — starting inference + bootstrap + plots"
"${PY}" "${SCRIPT}" --force --n-required "${N_REQUIRED}" 2>&1 | tee "${CFCRAFT}/mFrag_Figures/Figure6/detection_depth_figure6_run.log"
echo "[done]"
