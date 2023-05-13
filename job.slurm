#!/bin/bash
#SBATCH --job-name=controllora         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH -p gpu-ee
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends

#SBATCH --output=run_slurm/coco_only_more_layers.out    # name of output file


module purge
module load anaconda3/2023.3
conda activate controllora
accelerate launch  main.py --config configs/control_lora_laion_hed.yaml

#accelerate launch --multi_gpu --num_processes 2 main.py --config configs/control_lora_mpii_depth_encoder_only.yaml
# accelerate launch  main.py --config configs/control_lora_coco_canny.yaml
#python main.py --config configs/control_lora_laion_hed.yaml
#python examples/make_mpii_preprocessed.py
