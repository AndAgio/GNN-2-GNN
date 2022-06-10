#!/bin/bash
#SBATCH --job-name=generate-gnn2gnn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.agiollo@unibo.it
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /public/andrea.agiollo/venvs/gnn2gnn/bin/activate

python3 generate.py --train_dataset='nas101' --test_dataset='nas101' --nats_data='cifar10' --sub_skip='true' --sample_pre='true' --refine_gen_models='true' --complexity=1.0 --tau=0.1 --mu=2 --lmbd=0.1 --gen_mode='mine' --valuer_mode='class' --top_n_to_remove_from_dataset=30 --bench_dataset_folder='/public/andrea.agiollo/datasets/gnn2gnn/nas_benchmark_datasets' --dataset_folder='/public/andrea.agiollo/datasets/gnn2gnn/gnn2gnn_datasets' --out_folder='/public/andrea.agiollo/outputs/gnn2gnn' --out_path='/public/andrea.agiollo/outputs/gnn2gnn'