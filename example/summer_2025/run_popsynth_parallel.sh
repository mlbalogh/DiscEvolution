
#!/bin/bash
set -euo pipefail


PSI_VALUES="0.01 100"
MDOT_VALUES="1e-10 3e-10 1e-9 3e-9 1e-8 3e-8 1e-7 3e-7"
M_VALUES="0.01 0.05 0.1 0.125 0.15"
RD_VALUES="10 20 50 100 150 200"


NPROC=8
DRYRUN=0

COMPLETED_FILE="completed.txt"
LOGDIR="logs"
OUTDIR="/home/mbalogh/projects/PlanetFormation/DiscEvolution/output/HJpaper"

export DRYRUN LOGDIR OUTDIR COMPLETED_FILE
# Read completed.txt into a Bash associative array
declare -A DONE
while IFS=$'\t' read -r psi mdot M Rd rest; do
    [[ "$psi" == "psi" ]] && continue  # skip header
    key="$(printf "%.6g_%.6g_%.6g_%.6g" "$psi" "$mdot" "$M" "$Rd")"
    DONE["$key"]=1
done < completed.txt

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
      python3 run_model_stream.py --psi_DW "$psi" --Mdot "$mdot" --M "$M" --Rd "$Rd" \
        > "$LOGDIR/psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}.out" \
        2> "$LOGDIR/psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}.err"
    fi
  fi
' ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES


#!/bin/bash
# NPROC=8
# LOGDIR="logs"
# mkdir -p "$LOGDIR"
# OUTDIR="/home/mbalogh/projects/PlanetFormation/DiscEvolution/output/HJpaper"


# # Set DRYRUN=1 to only print what would be done
# DRYRUN=1
# format_psi () {
#   if [[ "$1" == "0.01" ]]; then
#     echo "0.01"
#   else
#     printf "%.1f" "$1"
#   fi
# }
# export -f format_psi

# parallel -j $([ "$DRYRUN" -eq 1 ] && echo 1 || echo $NPROC) --lb --tagstring 'psi{1}_Mdot{2}_M{3}_Rd{4}' \
#   '
#   OUTFILE="'"$OUTDIR"'/winds_mig_psi{=1 $_=q{`format_psi $_`} =}_Mdot{=2 $_=sprintf("%.1e",$_) =}_M{=3 $_=sprintf("%.1e",$_) =}_Rd{=4 $_=sprintf("%.1e",$_) =}.h5"

#   if [[ -f "$OUTFILE" ]]; then
#     echo "[`date +%F" "%T`] Skipping: $OUTFILE"
#   else
#     if [[ '"$DRYRUN"' -eq 1 ]]; then
#       echo "[`date +%F" "%T`] Would launch: $OUTFILE"
#     else
#       echo "[`date +%F" "%T`] Launching: $OUTFILE"
#       python3 run_model_stream.py --psi_DW {1} --Mdot {2} --M {3} --Rd {4} \
#         > "'"$LOGDIR"'/psi{1}_Mdot{2}_M{3}_Rd{4}.out" \
#         2> "'"$LOGDIR"'/psi{1}_Mdot{2}_M{3}_Rd{4}.err"
#     fi
#   fi
#   ' ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES




