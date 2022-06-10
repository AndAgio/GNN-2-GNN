import os
import argparse
# Import my modules
from scripts.train import GeneratorTrainer


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for training of NN generation using GNN.')
    parser.add_argument('--dataset', default='nas101',
                        help='dataset to be used between nas101')
    parser.add_argument('--nats_data', default='cifar10',
                        help='dataset to be used in nats selection between cifar10, cifar100, ImageNet16-120')
    parser.add_argument('--top_n_to_remove_from_dataset', default=10, type=int,
                        help='percentage of top models to remove from the dataset at hand')
    parser.add_argument('--sub_skip', default='False',
                        help='True if skip connection nodes are substituted in NATS')
    parser.add_argument('--gen_mode', default='mol_gan',
                        help='Generator model. Either \'mine\', \'mol_gan\' or \'rnn\'')
    parser.add_argument('--valuer_mode', default='class',
                        help='Valuer model. Either using classification \'class\' or regression \'reg\'')
    parser.add_argument('--sample_pre', default='false',
                        help='either sampling before or after the nodes function assignment')
    parser.add_argument('--lmbd', default=0.1, type=float,
                        help='balance factor between discriminator and regressor')
    parser.add_argument('--tau', default=1, type=float,
                        help='tau parameter for the gumbel softmax')
    parser.add_argument('--complexity', default=1, type=float,
                        help='complexity parameter for the dataset splitting')
    parser.add_argument('--mu', default=1, type=int,
                        help='parameter defining number of GCN layers of GNN')
    parser.add_argument('--refine_gen_models', default='true',
                        help='either if refinement procedure is used for generated graphs or not')
    parser.add_argument('--cio', default='false',
                        help='either if input output operations are predicted or not')
    parser.add_argument('--optimizer', default='adam',
                        help='optimizer to be used during training')
    parser.add_argument('--batch_size', default=32,
                        help='number of graphs to be used in each batch')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of epochs to be used in training')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate to be used in training')
    parser.add_argument('--lr_sched', default='plateau',
                        help='learning rate scheduler to be used in training')
    parser.add_argument('--lr_decay', default=0.1, type=float,
                        help='learning rate decay factor to be used in training')
    parser.add_argument('--lr_step_size', default=10, type=int,
                        help='step at which learning rate will decay'
                             ' or patience to be used for plateaus')
    parser.add_argument('--lr_sched_metric', default='loss',
                        help='metric to be used to update lr on plateau')
    parser.add_argument('--best_model_metric', default='top_10',
                        help='metric to check in order to store best regressor')
    parser.add_argument('--bench_dataset_folder', default='nas_benchmark_datasets',
                        help='folder where original benchmark datasets are stored')
    parser.add_argument('--dataset_folder', default='gnn2gnn_datasets',
                        help='folder where processed datasets are stored')
    parser.add_argument('--out_folder', default='outputs',
                        help='folder where outputs are stored (best trained models, etc.)')
    parser.add_argument('--selected_device', default='cpu',
                        help='cuda if GPU is needed, cpu otherwise')
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    gen_trainer = GeneratorTrainer(model_gen=args.gen_mode,
                                   model_dis='base',
                                   model_valuer=args.valuer_mode,
                                   dataset=args.dataset,
                                   nats_data=args.nats_data,
                                   top_n_to_remove_from_dataset=args.top_n_to_remove_from_dataset,
                                   sub_skip=False if args.sub_skip.lower() in ['false', 'f'] else True,
                                   sample_pre=False if args.sample_pre.lower() in ['false', 'f'] else True,
                                   tau=args.tau,
                                   complexity=args.complexity,
                                   mu=args.mu,
                                   refine_gen_models=False if args.refine_gen_models.lower() in ['false', 'f'] else True,
                                   cio=False if args.cio.lower() in ['false', 'f'] else True,
                                   optimizer=args.optimizer,
                                   batch_size=args.batch_size,
                                   loss_dis='crossentropy',
                                   loss_reg='crossentropy',  # 'mse',
                                   lmbd=args.lmbd,
                                   metrics=['acc', 'val', 'nov', 'uni', 'top_5', 'top_10', 'top_20', 'top_50', 'top_100'],
                                   lr=args.lr,
                                   epochs=args.epochs,
                                   lr_sched=args.lr_sched,
                                   lr_decay=args.lr_decay,
                                   lr_step_size=args.lr_step_size,
                                   lr_metric_to_check=args.lr_sched_metric,
                                   metric_to_check='top_10',
                                   bench_dataset_folder=args.bench_dataset_folder,
                                   dataset_folder=args.dataset_folder,
                                   out_path=args.out_folder,
                                   selected_device=args.selected_device)
    gen_trainer.run()


if __name__ == '__main__':
    args = gather_settings()
    main(args)
