#!/bin/sh
#BSUB -J modeldriven
#BSUB -q compute
#BSUB -W 40:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -M 10000MB
#BSUB -o ./lsf_output/modeldriven_%J_%I.out
#BSUB -e ./lsf_output/modeldriven_%J_%I.err
source ../../venv/bin/activate
python3 ./tests/clf_nn_mnist.py 
