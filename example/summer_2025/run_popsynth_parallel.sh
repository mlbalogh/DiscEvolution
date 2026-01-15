
#!/bin/bash
set -euo pipefail

# V1
# PSI_VALUES="0.01"
# MDOT_VALUES="1e-10 3e-10 1e-9 3e-9 1e-8 3e-8 1e-7 3e-7"
# M_VALUES="0.05 0.1 0.125 0.15"
# RD_VALUES="10 20 50 100 150 200"
# 
# V2
PSI_VALUES="0.01 10"
MDOT_VALUES="3e-9 1e-8 3e-8 1e-7 3e-7 1e-6"
M_VALUES="0.05 0.075 0.1 0.125 0.15"
RD_VALUES="20 50 100 150 200"
#
# Test
# PSI_VALUES="0.01"
# MDOT_VALUES="3e-8"
# M_VALUES="0.1"
# RD_VALUES="50 100"


NPROC=8
DRYRUN=0

COMPLETED_FILE="completed.txt"
LOGDIR="logs"
# Use DISCEVOLUTION_OUTPUT env var if set, otherwise fall back to default
OUTDIR="${DISCEVOLUTION_OUTPUT:-.}"
# Get absolute path to script directory and config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/DiscConfig_HJpaper.json"

export DRYRUN LOGDIR OUTDIR COMPLETED_FILE SCRIPT_DIR CONFIG_FILE
# Read COMPLETED_FILE into a Bash associative array
declare -A DONE
while IFS=$'\t' read -r psi mdot M Rd rest; do
    [[ "$psi" == "psi" ]] && continue  # skip header
    key="$(printf "%.6g_%.6g_%.6g_%.6g" "$psi" "$mdot" "$M" "$Rd")"
    DONE["$key"]=1
done < ${COMPLETED_FILE}

# Export the associative array so subshells see it
export LOGDIR OUTDIR DRYRUN
declare -p DONE > /tmp/done_array.sh

JOBS=$([ "$DRYRUN" -eq 1 ] && echo 1 || echo "$NPROC")

parallel -j "$JOBS" --lb --tagstring 'psi{1}_Mdot{2}_M{3}_Rd{4}' '
  source /tmp/done_array.sh
  psi={1}; mdot={2}; M={3}; Rd={4}
  key="$(printf "%.6g_%.6g_%.6g_%.6g" "$psi" "$mdot" "$M" "$Rd")"
  if [[ -n "${DONE[$key]:-}" ]]; then
    echo "Skipping: $key"
  else
    psi_fmt="$psi"; [[ "$psi" != *.* ]] && psi_fmt="${psi}.0"
    mdot_fmt=$(printf "%.1e" "$mdot")
    M_fmt=$(printf "%.1e" "$M")
    Rd_fmt=$(printf "%.1e" "$Rd")
    OUTFILE="$OUTDIR/winds_mig_psi${psi_fmt}_Mdot${mdot_fmt}_M${M_fmt}_Rd${Rd_fmt}.h5"

    if [[ $DRYRUN -eq 1 ]]; then
      echo "[`date +%F" "%T`] Would launch: $OUTFILE"
    else
      echo "[`date +%F" "%T`] Launching: $OUTFILE"
      python3 run_model_discchem_stream.py --config "$CONFIG_FILE"  --psi_DW "$psi" --Mdot "$mdot" --M "$M" --Rd "$Rd" \
        > "$LOGDIR/winds_mig_psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}.out" \
        2> "$LOGDIR/winds_mig_psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}.err"
    fi
  fi
' ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES





