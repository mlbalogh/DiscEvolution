#!/bin/bash
PSI_VALUES="0.01 1 100"
MDOT_VALUES="1e-10 3e-10 1e-9 3e-9 1e-8 3e-8 1e-7 3e-7"
M_VALUES="0.01 0.05 0.1 0.125 0.15"
RD_VALUES="10 20 50 100 150 200"

OUTDIR="/home/mbalogh/projects/PlanetFormation/DiscEvolution/output/HJpaper"

parallel --tagstring 'psi{1}_Mdot{2}_M{3}_Rd{4}' echo '
  #OUTFILE='"$OUTDIR"'/winds_mig_{=1 $_=sprintf("%.2g",$_) =}_Mdot{2}_M{3}_Rd{4}.h5
  OUTFILE="'"$OUTDIR"'/winds_mig_psi{1}_Mdot{=2 $_=sprintf("%.1e",$_) =}_M{=3 $_=sprintf("%.1e",$_) =}_Rd{=4 $_=sprintf("%.1e",$_) =}.h5"
  if [[ -f $OUTFILE ]]; then
      echo "Skipping: $OUTFILE"
  else
      echo "Would launch: $OUTFILE"
  fi
' ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES
