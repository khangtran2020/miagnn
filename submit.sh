#!/bin/bash -l

#SBATCH -J nodeDP  #job name
#SBATCH -o results/logs/blackbox.out
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:1    #number of gpus needed, default is 0
#SBATCH -c 1            #number of CPUs needed, default is 1
#SBATCH --mem=48G    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=tkhang@hbku.edu.qa

module load cuda11.3/toolkit/11.3.0
conda activate nodedp

for data in cora citeseer facebook
do
    for gen_sub in ind trans
    do
        for att_sub in joint sep
        do
            for run in 1 2 3
            do
                python main.py  --proj_name "blackbox-${data}-${gen_sub}-${att_sub}" \
                        --gen_mode clean \
                        --gen_submode $gen_sub \
                        --data $data \
                        --data_mode none \
                        --model sage \
                        --bs 512 \
                        --nnei 20 \
                        --lr 0.01 \
                        --opt adam \
                        --nlay 2 \
                        --hdim 32 \
                        --epochs 200 \
                        --att_mode blackbox \
                        --att_submode $att_sub \
                        --att_epochs 200 \
                        --att_lr 0.0001 \
                        --att_bs 512 \
                        --att_hdim 64 \
                        --att_lay 3 \
                        --sha_rat 0.2 \
                        --sha_epochs 200\
                        --retrain 0 \
                        --device gpu \
                        --seed $run
            done
        done
    done
done
