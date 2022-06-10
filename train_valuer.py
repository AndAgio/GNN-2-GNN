import os
import argparse
# Import my modules
from scripts.train import ValuerTrainer


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for training of NN generation using GNN.')
    parser.add_argument('--dataset', default='nas101',
                        help='dataset to be used between nas101')
    parser.add_argument('--nats_data', default='cifar10',
                        help='dataset to be used in nats selection between cifar10, cifar100, ImageNet16-120')
    parser.add_argument('--sub_skip', default='False',
                        help='True if skip connection nodes are substituted in NATS')
    parser.add_argument('--valuer', default='base',
                        help='valuer to use for segmentation')
    parser.add_argument('--is_reg', default='false',
                        help='True if pure regression on accuracy is used. False if classification over model goodness is used.')
    parser.add_argument('--sample_pre', default='false',
                        help='either sampling before or after the nodes function assignment')
    parser.add_argument('--lmbd', default=0.01, type=float,
                        help='balance factor between discriminator and valuer')
    parser.add_argument('--tau', default=1, type=float,
                        help='tau parameter for the gumbel softmax')
    parser.add_argument('--complexity', default=1, type=float,
                        help='complexity parameter for the dataset splitting')
    parser.add_argument('--mu', default=1, type=int,
                        help='parameter defining number of GCN layers of GNN')
    parser.add_argument('--refine_gen_models', default='true',
                        help='either if refinement procedure is used for generated graphs or not')
    parser.add_argument('--cio', default='true',
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
    reg_trainer = ValuerTrainer(dataset=args.dataset,
                                nats_data=args.nats_data,
                                sub_skip=False if args.sub_skip.lower() in ['false', 'f'] else True,
                                complexity=args.complexity,
                                model=args.valuer,
                                optimizer=args.optimizer,
                                batch_size=args.batch_size,
                                is_reg=False if args.is_reg.lower() in ['false', 'f'] else True,
                                lr=args.lr,
                                epochs=args.epochs,
                                lr_sched=args.lr_sched,
                                lr_decay=args.lr_decay,
                                lr_step_size=args.lr_step_size,
                                bench_dataset_folder=args.bench_dataset_folder,
                                dataset_folder=args.dataset_folder,
                                out_path=args.out_folder)
    reg_trainer.run()


if __name__ == '__main__':
    args = gather_settings()
    main(args)
