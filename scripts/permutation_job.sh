#!/bin/sh
#BSUB -J modeldriven_s0_b64_a50
#BSUB -q compute
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -M 10000MB
###BSUB -u polgarciarecasens@gmail.com
###BSUB -Ne 
#BSUB -o ./lsf_output/modeldriven_s0_b64_a50__%J_%I.out
#BSUB -e ./lsf_output/modeldriven_s0_b64_a50__%J_%I.err
source ../../venv/bin/activate
python ./tests/clf_nn_pinwheel.py 