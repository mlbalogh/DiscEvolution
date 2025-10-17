#!/bin/bash
# run_models.sh
# Sweep parameters and run up to 8 simulations in parallel.

PSI_VALUES="0.01 1 100"
MDOT_VALUES="1e-10 3e-10 1e-9 3e-9 1e-8 3e-8 1e-7 3e-7"
M_VALUES="0.01 0.05 0.1 0.125 0.15"
RD_VALUES="10 20 50 100 150 200"

NPROC=8
count=0

LOGDIR="logs"
mkdir -p "$LOGDIR"

timestamp() {
  date +"[%Y-%m-%d %H:%M:%S]"
}

for psi in $PSI_VALUES; do
  for mdot in $MDOT_VALUES; do
    for m in $M_VALUES; do
      for rd in $RD_VALUES; do
        tag="psi${psi}_Mdot${mdot}_M${m}_Rd${rd}"
        echo "$(timestamp) Launching run: $tag"

        python run_model_single.py --psi_DW "$psi" --Mdot "$mdot" --M "$m" --Rd "$rd" \
          > "${LOGDIR}/${tag}.out" 2> "${LOGDIR}/${tag}.err" &

        ((count++))
        if ((count % NPROC == 0)); then
          echo "$(timestamp) Reached $NPROC concurrent jobs, waiting..."
          wait
        fi
      done
    done
  done
done

wait
echo "$(timestamp) All simulations complete."

