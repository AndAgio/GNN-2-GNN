import os
import argparse
import random
from collections import Counter
import numpy as np
import torch
from progress.bar import IncrementalBar
# Import my modules
from scripts.data import NASDataset2Splits as NASDataset
from scripts.data import NATSDataset2Splits as NATSDataset
from scripts.metrics import Validity, InTopKPercent, AccVsFootprint, GeneratorMetrics, Novelty, ModelAccuracy
from scripts.random_baseline import RandomGenerator
from scripts.runner import GeneratorRunner
from scripts.utils import scatter_plot_acc_vs_footprint, Utils, convert_edges_to_adj, hash_model, \
    bar_plot_acc_vs_footprint


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for generating NN using our GNN.')
    parser.add_argument('--gen_mode', default='mine', nargs='+',
                        help='generator model used to generate graphs')
    parser.add_argument('--train_dataset', default='nas101',
                        help='dataset over which model is trained between nas101 and nats')
    parser.add_argument('--test_dataset', default='nas101',
                        help='dataset to be used for testing performance between nas101 and nats')
    parser.add_argument('--nats_data', default='cifar10',
                        help='dataset to be used in nats selection between cifar10, cifar100, ImageNet16-120')
    parser.add_argument('--top_n_to_remove_from_dataset', default=10, type=int,
                        help='percentage of top models to remove from the dataset at hand')
    parser.add_argument('--valuer_mode', default='class',
                        help='Valuer model. Either using classification \'class\' or regression \'reg\'')
    parser.add_argument('--dataset_folder', default='gnn2gnn_datasets',
                        help='folder where processed datasets are stored')
    parser.add_argument('--bench_dataset_folder', default='nas_benchmark_datasets',
                        help='folder where original benchmark datasets are stored')
    parser.add_argument('--out_path', default='outputs',
                        help='path where output files are stored during training')
    parser.add_argument('--sub_skip', default='true',
                        help='True if skip connections need to be substituted in NATS')
    parser.add_argument('--sample_pre', default='false',
                        help='either sampling before or after the nodes function assignment')
    parser.add_argument('--tau', default=1, type=float, nargs='+',
                        help='tau parameter for the gumbel softmax')
    parser.add_argument('--complexity', default=1, type=float,
                        help='complexity parameter for the dataset splitting')
    parser.add_argument('--mu', default=1, type=int, nargs='+',
                        help='parameter defining number of GCN layers of GNN')
    parser.add_argument('--lmbd', default=0.01, type=float, nargs='+',
                        help='balance factor between discriminator and regressor')
    # parser.add_argument('--n_nodes', default=7, type=int,
    #                     help='number of nodes in the graph')
    parser.add_argument('--refine_gen_models', default='true',
                        help='either if randomly generated graphs are refined or not')
    parser.add_argument('--cio', default='true',
                        help='either if input output operations are predicted or not')
    parser.add_argument('--n_graphs', default=1000, type=int,
                        help='number of graphs to be generated')
    parser.add_argument('--out_folder', default='test',
                        help='folder where generated models are stored')
    args = parser.parse_args()
    return args


def main(args):
    # Import dataset
    print('Importing dataset...')
    if args.test_dataset == 'nas101':
        train_dataset = NASDataset(root=os.path.join(args.dataset_folder, 'NAS101'),
                                   bench_folder=os.path.join(args.bench_dataset_folder, 'NAS101'),
                                   top_n=args.top_n_to_remove_from_dataset)
    elif args.test_dataset == 'nats':
        train_dataset = NATSDataset(root=os.path.join(args.dataset_folder, 'NATS'),
                                    bench_folder=os.path.join(args.bench_dataset_folder, 'NATS'),
                                    sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                    top_n=args.top_n_to_remove_from_dataset,
                                    chosen_data=args.nats_data)
    else:
        raise ValueError('Not a valid dataset!')
    n_nodes = train_dataset.get_num_nodes()
    # Define metrics
    metrics = {'val': None,
               'nov': None,
               'top_5': None,
               'top_10': None,
               'top_20': None,
               'top_50': None,
               'top_100': None,
               'acc_vs_foot': None, }
    gen_metrics = GeneratorMetrics(train_dataset)
    dataset_metrics = gen_metrics.get_dataset_metrics()
    train_models_hashes = gen_metrics.get_train_models_hashes()
    for metric in metrics.keys():
        if metric == 'val':
            metrics[metric] = Validity()
        elif metric == 'nov':
            metrics[metric] = Novelty(train_models_hashes=train_models_hashes)
        elif metric[:3] == 'top':
            metrics[metric] = InTopKPercent(dataset_metrics=dataset_metrics, top_k=int(metric.split('_')[-1]))
        elif metric == 'acc_vs_foot':
            metrics[metric] = AccVsFootprint(dataset_metrics=dataset_metrics)
    # Define random generator
    random_generator = RandomGenerator(n_nodes=n_nodes, dataset=args.test_dataset,
                                       sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                       cio=True if args.cio.lower() in ['true', 't'] else False)
    # Run the generator multiple times and evaluate the metrics
    scores_random = random_generator.run(metrics=metrics,
                                         n_graphs=args.n_graphs,
                                         refine=True if args.refine_gen_models.lower() in ['true', 't'] else False)
    acc_vs_foot_random = [acc_vs_foot for acc_vs_foot in scores_random['acc_vs_foot'] if acc_vs_foot != [None, None]]
    # Get acc vs footprint for graphs in dataset
    acc_vs_foot_dataset = metrics['acc_vs_foot'].get_all_acc_vs_footprint()
    acc_vs_foot_dataset = random.sample(acc_vs_foot_dataset, int(len(acc_vs_foot_dataset)/10))
    # Print average accuracy of predicted NN architectures
    if args.test_dataset == 'nats':
        accs = [item[0] for item in acc_vs_foot_random]
    else:
        accs = [item[0] * 100 for item in acc_vs_foot_random]
    print('Average accuracy: {:.4f}%'.format(np.mean(accs)))
    print('Best accuracy: {:.4f}%'.format(np.max(accs)))
    print('Number of models with accuracy >= 90%: {:.4f}%'.format(100*len([True for a in accs if a >= 90])/float(len(accs))))
    print('Number of models with accuracy >= 91%: {:.4f}%'.format(100*len([True for a in accs if a >= 91])/float(len(accs))))
    print('Number of models with accuracy >= 92%: {:.4f}%'.format(100*len([True for a in accs if a >= 92])/float(len(accs))))
    print('Number of models with accuracy >= 93%: {:.4f}%'.format(100*len([True for a in accs if a >= 93])/float(len(accs))))
    print('Number of models with accuracy >= 94%: {:.4f}%'.format(100*len([True for a in accs if a >= 94])/float(len(accs))))
    foots = [item[1] for item in acc_vs_foot_random]
    print('Average footprint: {} {}'.format(np.mean(foots), 'MB' if args.test_dataset == 'nats' else 'parameters'))
    # Define initial dictionary for the final plot
    # Define dictionary for scatter plotting
    if len(list(set(args.gen_mode))) == 1:
        acc_vs_foot_dict = {args.test_dataset.upper(): acc_vs_foot_dataset, }
    else:
        acc_vs_foot_dict = {args.test_dataset.upper(): acc_vs_foot_dataset,
                            'random'.upper(): acc_vs_foot_random, }
    # Iterate over list of parameters mu, tau, lambda
    assert len(args.gen_mode) == len(args.mu) == len(args.tau) == len(args.lmbd)
    for index in range(len(args.mu)):
        # Get parameter from list
        gen_mode = args.gen_mode[index]
        mu = args.mu[index]
        tau = args.tau[index]
        lmbd = args.lmbd[index]
        # Load generator from trained models
        generator = load_generator(args, gen_mode, mu, tau, lmbd)
        # Import generator runner
        generator_runner = GeneratorRunner(generator,
                                           n_nodes=n_nodes,
                                           dataset=args.test_dataset,
                                           generator_name='{}_to_{}'.format(args.train_dataset, args.test_dataset),
                                           sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                           out_path=args.out_path)
        # Run the generator multiple times and evaluate the metrics
        scores_generator, gen_hashes_list = generator_runner.run(metrics=metrics,
                                                n_graphs=args.n_graphs,
                                                plot=False)
        acc_vs_foot_gen = [acc_vs_foot for acc_vs_foot in scores_generator['acc_vs_foot'] if
                           acc_vs_foot != [None, None]]
        # print('acc_vs_foot_gen: {}'.format(acc_vs_foot_gen))
        # title = r'$GNN2GNN_{\mu = ' + str(mu) + ', \tau = ' + str(tau) + ', \lambda = ' + str(lmbd) + '}$'
        # title = r'$GNN2GNN_{\lambda = ' + str(lmbd) + '}$'
        title = get_title_from_variables(args, index)
        acc_vs_foot_dict[title] = acc_vs_foot_gen
        # acc_vs_foot_dict[r'$GNN2GNN_{\mu =' + str(mu) + ',\n\tau =' + str(tau) + ',\n\lambda =' + str(lmbd) + '}$'] = acc_vs_foot_gen
        # Print average accuracy of predicted NN architectures
        if args.test_dataset == 'nats':
            accs = [item[0] for item in acc_vs_foot_gen]
        else:
            accs = [item[0] * 100 for item in acc_vs_foot_gen]
        print('Average accuracy: {:.4f}%'.format(np.mean(accs)))
        print('Best accuracy: {:.4f}%'.format(np.max(accs)))
        print('Number of models with accuracy >= 90%: {:.4f}%'.format(100*len([True for a in accs if a >= 90])/float(len(accs))))
        print('Number of models with accuracy >= 91%: {:.4f}%'.format(100*len([True for a in accs if a >= 91])/float(len(accs))))
        print('Number of models with accuracy >= 92%: {:.4f}%'.format(100*len([True for a in accs if a >= 92])/float(len(accs))))
        print('Number of models with accuracy >= 93%: {:.4f}%'.format(100*len([True for a in accs if a >= 93])/float(len(accs))))
        print('Number of models with accuracy >= 94%: {:.4f}%'.format(100*len([True for a in accs if a >= 94])/float(len(accs))))
        foots = [item[1] for item in acc_vs_foot_gen]
        print('Average footprint: {} {}'.format(np.mean(foots), 'MB' if args.test_dataset == 'nats' else 'parameters'))
        # Compute uniqueness score from list of generated graphs
        counter_dict = Counter(gen_hashes_list)
        n_uniques = 0
        for k, v in counter_dict.items():
            if v == 1:
                n_uniques += 1
        print('Number of unique graphs: {}'.format(float(n_uniques)/len(gen_hashes_list)))
    # Plot
    # Define output folder for storing the plot in pdf format
    out_path = os.path.join(args.out_path, 'gnn2gnn_generated_plots', '{}_to_{}'.format(args.train_dataset, args.test_dataset))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    scatter_plot_acc_vs_footprint(acc_vs_foot_dict, dataset=args.test_dataset, out_path=out_path)
    # bar_plot_acc_vs_footprint(acc_vs_foot_dict, dataset=args.test_dataset, out_path=out_path)


def new_main(args):
    # Import dataset
    print('Importing dataset...')
    nas101_train_dataset = NASDataset(root=os.path.join(args.dataset_folder, 'NAS101'),
                                   bench_folder=os.path.join(args.bench_dataset_folder, 'NAS101'),
                                   top_n=args.top_n_to_remove_from_dataset)
    nats_train_dataset = NATSDataset(root=os.path.join(args.dataset_folder, 'NATS'),
                                    bench_folder=os.path.join(args.bench_dataset_folder, 'NATS'),
                                    sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                    top_n=args.top_n_to_remove_from_dataset,
                                    chosen_data=args.nats_data)
    n_nodes = nas101_train_dataset.get_num_nodes()
    # Define metrics
    gen_metrics = GeneratorMetrics(nas101_train_dataset)
    dataset_metrics = gen_metrics.get_dataset_metrics()
    nas101_top_10_metric = InTopKPercent(dataset_metrics=dataset_metrics, top_k=10)
    nas101_top_100_metric = InTopKPercent(dataset_metrics=dataset_metrics, top_k=100)
    nas101_acc = ModelAccuracy(dataset_metrics=dataset_metrics, is_reg=True)
    gen_metrics = GeneratorMetrics(nats_train_dataset)
    dataset_metrics = gen_metrics.get_dataset_metrics()
    nats_top_10_metric = InTopKPercent(dataset_metrics=dataset_metrics, top_k=10)
    nats_top_100_metric = InTopKPercent(dataset_metrics=dataset_metrics, top_k=100)
    nats_acc = ModelAccuracy(dataset_metrics=dataset_metrics, is_reg=True)
    # Define random generator
    # random_generator = RandomGenerator(n_nodes=n_nodes, dataset=args.test_dataset,
    #                                    sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
    #                                    cio=True if args.cio.lower() in ['true', 't'] else False)
    # # Run the generator multiple times and evaluate the metrics
    # _, gen_graph_loader = random_generator.get_single_graph(refine=True if args.refine_gen_models.lower() in ['true', 't'] else False)
    # graph = iter(gen_graph_loader).next()
    # Import trained generator and use it to generate models
    generator = load_generator(args, gen_mode='mine', mu=2, tau=0.1, lmbd=0.1)
    # Import generator runner
    generator_runner = GeneratorRunner(generator,
                                       n_nodes=n_nodes,
                                       dataset=args.test_dataset,
                                       generator_name='{}_to_{}'.format(args.train_dataset, args.test_dataset),
                                       sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                       out_path=args.out_path)
    # Run the generator to get a single graph
    _, gen_graph_loader = generator_runner.get_single_graph()
    graph = iter(gen_graph_loader).next()
    # Compute score over single graph and add it to running scores
    print('NAS101 top-10 metric: {}'.format(nas101_top_10_metric.compute(graph)))
    print('NATS top-10 metric: {}'.format(nats_top_10_metric.compute(graph)))
    print('NAS101 top-100 metric: {}'.format(nas101_top_100_metric.compute(graph)))
    print('NATS top-100 metric: {}'.format(nats_top_100_metric.compute(graph)))
    print('NAS101 model accuracy: {}'.format(nas101_acc.compute(graph)))
    print('NATS model accuracy: {}'.format(nats_acc.compute(graph)))
    print('GRAPH:\nX = {}\nEDGES = {}'.format(graph.x.float(), graph.edge_index))
    Utils().plot_tg_data(graph, sub_skip=True)


def compare_nats_and_nas101(args):
    # Import dataset
    print('Importing dataset...')
    nas101_dataset = NASDataset(root=os.path.join(args.dataset_folder, 'NAS101'),
                                bench_folder=os.path.join(args.bench_dataset_folder, 'NAS101'),
                                top_n=args.top_n_to_remove_from_dataset)
    nats_dataset = NATSDataset(root=os.path.join(args.dataset_folder, 'NATS'),
                               bench_folder=os.path.join(args.bench_dataset_folder, 'NATS'),
                               sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                               top_n=args.top_n_to_remove_from_dataset,
                               chosen_data=args.nats_data)
    n_nodes = nas101_dataset.get_num_nodes()
    # Define metrics
    gen_metrics = GeneratorMetrics(nas101_dataset)
    nas101_dataset_metrics = gen_metrics.get_dataset_metrics()
    nas101_top_10_metric = InTopKPercent(dataset_metrics=nas101_dataset_metrics, top_k=10)
    nas101_top_100_metric = InTopKPercent(dataset_metrics=nas101_dataset_metrics, top_k=100)
    nas101_acc = ModelAccuracy(dataset_metrics=nas101_dataset_metrics, is_reg=True)
    gen_metrics = GeneratorMetrics(nats_dataset)
    nats_dataset_metrics = gen_metrics.get_dataset_metrics()
    nats_top_10_metric = InTopKPercent(dataset_metrics=nats_dataset_metrics, top_k=10)
    nats_top_100_metric = InTopKPercent(dataset_metrics=nats_dataset_metrics, top_k=100)
    nats_acc = ModelAccuracy(dataset_metrics=nats_dataset_metrics, is_reg=True)
    # Get all models from the NATS dataset
    nas101_models = nas101_dataset.get_all_dataset_models()
    nats_models = nats_dataset.get_all_dataset_models()
    # Iterate over all NATS models
    absolute_difference = 0
    shared_models = 0
    top10_in_nats = 0
    top10_in_nas101 = 0
    shared_top_10 = 0
    bar = IncrementalBar('Getting the accuracy MAE...', max=len(nats_models))
    for model in nats_models:
        # print('model: {}'.format(model))
        x, edge_index = model.x, model.edge_index
        # Convert edges to adj_mat for faster computation
        adj_mat = convert_edges_to_adj(x, edge_index)
        model_acc_in_nats = model.test_accuracy
        nats_top = model.top
        print('nats_top: {}'.format(nats_top))
        if nats_top < 10:
            nats_in_top_10 = True
            top10_in_nats += 1
        else:
            nats_in_top_10 = False
        # print('model_acc_in_nats: {}'.format(model_acc_in_nats))
        # print('model_acc_in_nats: {}'.format(nats_acc.get_model_acc(x, adj_mat)[0]*100))
        # model_acc_in_nas101, _ = nas101_acc.get_model_acc(x, adj_mat)
        # Compute hash for single model
        model_hash = hash_model(x, adj_mat)
        # Try to get model metrics from the dictionary of metrics of the dataset
        try:
            model_metrics = nas101_dataset_metrics[model_hash]
            model_acc_in_nas101 = model_metrics['test_accuracy']
            model_acc_in_nas101 *= float(100)
            # print('model_acc_in_nas101: {}'.format(model_acc_in_nas101))
            absolute_difference += abs(model_acc_in_nats - model_acc_in_nas101)
            shared_models += 1

            nas101_top = model_metrics['top']
            if nas101_top < 10:
                nas101_in_top_10 = True
                top10_in_nas101 += 1
            else:
                nas101_in_top_10 = False
            if nas101_in_top_10 and nats_in_top_10:
                shared_top_10 += 1
        except KeyError:
            pass
        bar.next()
    bar.finish()
    print('Number of models in common between NATS and NAS101: {}'.format(shared_models))
    mae = absolute_difference/float(shared_models)
    print('Accuracy MAE between NATS and NAS101: {:.3f}%'.format(mae))
    print('Total number of models in NATS: {} \t Top 10 in NATS: {}'.format(len(nats_models), top10_in_nats))
    print('Total number of models in NAS101: {} \t Top 10 in NAS101: {}'.format(len(nas101_models), top10_in_nas101))
    print('Models sharing the top 10 in NAS101 and NATS: {}'.format(shared_top_10))

    nas101_loader = nas101_dataset.get_data_loader(batch_size=1, shuffle=False)
    tot_acc = 0
    for _, data in enumerate(nas101_loader, 0):
        # print('nas101 top: {}'.format(data.top))
        tot_acc += data.test_accuracy.numpy()
    print('Average accuracy for training set of NAS101: {}'.format(tot_acc/float(len(nas101_dataset))))

    nats_loader = nats_dataset.get_data_loader(batch_size=1, shuffle=False)
    tot_acc = 0
    for _, data in enumerate(nats_loader, 0):
        # print('nats top: {}'.format(data.top))
        tot_acc += data.test_accuracy.numpy()
    print('Average accuracy for training set of NATS: {}'.format(tot_acc/float(len(nats_dataset))))


def plot_nats_and_nas101(args):
    # Import dataset
    print('Importing dataset...')
    nas101_dataset = NASDataset(root=os.path.join(args.dataset_folder, 'NAS101'),
                                bench_folder=os.path.join(args.bench_dataset_folder, 'NAS101'),
                                top_n=args.top_n_to_remove_from_dataset)
    nats_dataset = NATSDataset(root=os.path.join(args.dataset_folder, 'NATS'),
                               bench_folder=os.path.join(args.bench_dataset_folder, 'NATS'),
                               sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                               top_n=args.top_n_to_remove_from_dataset,
                               chosen_data=args.nats_data)

    utils = Utils()
    nas101_loader = nas101_dataset.get_data_loader(batch_size=1, shuffle=False)
    tot_acc = 0
    for index, data in enumerate(nas101_loader, 0):
        # print('nas101 top: {}'.format(data.top))
        tot_acc += data.test_accuracy.numpy()
        utils.plot_tg_data_to_store(data,
                                    store_path=os.path.join('outputs', 'NAS101_plots', '{}.pdf'.format(index)),
                                    bench='nas101',
                                    title='Acc: {:.2f}%  Top-{:.2f}%'.format(data.test_accuracy.numpy()[0]*100,
                                                                                  data.top.numpy()[0]))
    print('Average accuracy for training set of NAS101: {}'.format(tot_acc/float(len(nas101_dataset))))

    nats_loader = nats_dataset.get_data_loader(batch_size=1, shuffle=False)
    tot_acc = 0
    for index, data in enumerate(nats_loader, 0):
        # print('nats top: {}'.format(data.top))
        tot_acc += data.test_accuracy.numpy()
        utils.plot_tg_data_to_store(data,
                                    store_path=os.path.join('outputs', 'NATS_plots', '{}.pdf'.format(index)),
                                    bench='nats',
                                    sub_skip=True if args.sub_skip.lower() in ['true', 't'] else False,
                                    title='Acc: {:.2f}%  Top-{:.2f}%'.format(data.test_accuracy.numpy()[0],
                                                                                  data.top.numpy()[0]))
    print('Average accuracy for training set of NATS: {}'.format(tot_acc/float(len(nats_dataset))))


def load_generator(args, gen_mode, mu, tau, lmbd):
    print('Loading generator model...')
    if args.train_dataset == 'nats':
        out_folder_name = '{}_{}_{}_ss_{}_sp_{}_topdata_{}_vmode_{}_tau_{}_comp_{}_mu_{}_lmbd_{}'.format(gen_mode,
                                                                                                         args.train_dataset,
                                                                                                         args.nats_data,
                                                                                                         True if args.sub_skip.lower() in [
                                                                                                             'true',
                                                                                                             't'] else False,
                                                                                                         True if args.sample_pre.lower() in [
                                                                                                             'true',
                                                                                                             't'] else False,
                                                                                                         args.top_n_to_remove_from_dataset,
                                                                                                         args.valuer_mode,
                                                                                                         tau,
                                                                                                         args.complexity,
                                                                                                         mu,
                                                                                                         lmbd)
    elif args.train_dataset == 'nas101':
        out_folder_name = '{}_{}_sp_{}_topdata_{}_vmode_{}_tau_{}_comp_{}_mu_{}_lmbd_{}'.format(gen_mode,
                                                                                                args.train_dataset,
                                                                                                True if args.sample_pre.lower() in [
                                                                                                    'true',
                                                                                                    't'] else False,
                                                                                                args.top_n_to_remove_from_dataset,
                                                                                                args.valuer_mode,
                                                                                                tau,
                                                                                                args.complexity,
                                                                                                mu,
                                                                                                lmbd)
    trained_models_folder = os.path.join(args.out_path, out_folder_name, 'trained_models')
    print('trained_models_folder: {}'.format(trained_models_folder))
    model_name = 'gen_{}_{}_best.pt'.format(gen_mode, args.train_dataset)
    model_path = os.path.join(trained_models_folder, model_name)
    generator = torch.load(model_path)
    return generator


def get_title_from_variables(args, index):
    title = r'$'
    # Get model name
    title += get_model_name_from_arg(args.gen_mode[index])
    pedex = ''
    # Check if all mu values are the same
    if len(list(set(args.mu))) != 1:
        pedex += '\mu = ' + str(args.mu[index]) + ', '
    # Check if all tau values are the same
    if len(list(set(args.tau))) != 1:
        pedex += '\tau = ' + str(args.tau[index]) + ', '
    # Check if all lambda values are the same
    if len(list(set(args.lmbd))) != 1:
        pedex += '\lambda = ' + str(args.lmbd[index])
    title += '_{' + pedex + '}$'
    return title


def get_model_name_from_arg(name):
    if name == 'mine':
        return 'GNN2GNN'
    elif name == 'mol_gan':
        return 'MOLGAN'
    elif name == 'rnn':
        return name.upper()
    else:
        raise ValueError('No such name should be allowed in args for the generator model!')


if __name__ == '__main__':
    args = gather_settings()
    main(args)
    # compare_nats_and_nas101(args)
    # plot_nats_and_nas101(args)
