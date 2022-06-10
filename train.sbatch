#!/bin/bash
#SBATCH --job-name=train-gnn2gnn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.agiollo@unibo.it
#SBATCH --time=15-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /public/andrea.agiollo/venvs/gnn2gnn/bin/activate

python3 train_generator.py --selected_device='cuda' --dataset='nas101' --sub_skip='true' --sample_pre='true' --refine_gen_models='true' --complexity=1.0 --tau=0.1 --mu=1  --lmbd=1 --optimizer='adam' --batch_size=32 --lr=0.001 --epochs=20 --best_model_metric='top_10' --gen_mode='mine' --valuer_mode='class' --bench_dataset_folder='/public/andrea.agiollo/datasets/gnn2gnn/nas_benchmark_datasets' --dataset_folder='/public/andrea.agiollo/datasets/gnn2gnn/gnn2gnn_datasets' --out_folder='/public/andrea.agiollo/outputs/gnn2gnn'