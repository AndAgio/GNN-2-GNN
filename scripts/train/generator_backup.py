import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
import torch_geometric as tg
import networkx as nx
import matplotlib.pyplot as plt
# Import my modules
from scripts.data import SyntheticDataset
from scripts.data import NASDataset2Splits as NASDataset
from scripts.models import DiscriminatorNet, GeneratorNet
from scripts.metrics import Accuracy, Validity, Novelty, InTopKPercent
from scripts.utils import Utils, Logger, timeit


class GeneratorTrainer():
    def __init__(self, model_gen, model_dis, dataset,
                 n_nodes=7, hidden_dim=10,
                 optimizer='SGD', momentum=0.9,
                 weight_decay=5e-4, loss='bce',
                 metrics=['accuracy'], batch_size=32, epochs=100, lr=0.01,
                 lr_sched=None, lr_decay=None, lr_step_size=None,
                 lr_metric_to_check='loss',
                 metric_to_check='acc',
                 bench_dataset_folder='nas_benchmark_datasets',
                 dataset_folder='gnn2gnn_datasets',
                 out_path='outputs'):
        self.chosen_model_gen = model_gen
        self.chosen_model_dis = model_dis
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.chosen_optimizer = optimizer
        self.batch_size = batch_size if batch_size is None else int(batch_size)
        self.chosen_loss = loss
        self.chosen_metrics = metrics
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.chosen_scheduler = lr_sched
        if lr_sched is not None:
            assert lr_decay is not None
            self.lr_decay = lr_decay
            assert lr_step_size is not None
            self.lr_step_size = lr_step_size
            if lr_sched == 'plateau':
                self.lr_metric_to_check = lr_metric_to_check
        self.metric_to_check = metric_to_check
        self.bench_dataset_folder = bench_dataset_folder
        self.dataset_folder = dataset_folder
        self.out_path = os.path.join(out_path, 'generation')
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get dataset and setup the trainer
        self.get_dataset(dataset)
        self.setup()

    def get_dataset(self, dataset):
        print('Importing dataset...')
        self.chosen_dataset = dataset
        # Get the dataset depending on the selected one and move it to the torch device
        if dataset == 'nas101':
            # nas = NASDataset()
            self.train_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                            split='train')
            self.test_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                           split='test')
            self.train_loader = self.train_dataset.get_data_loader(batch_size=self.batch_size,
                                                                   shuffle=True)
            self.test_loader = self.test_dataset.get_data_loader(batch_size=self.batch_size,
                                                                 shuffle=False)
            # Get the number of node features and set number of classes
            self.num_node_features = self.train_dataset.num_features
            self.num_classes = self.train_dataset.num_classes
            # Empty folder containing synthetic graphs
            synthetic_path = os.path.join(self.dataset_folder, 'synthetic')
            if os.path.exists(synthetic_path) and os.path.isdir(synthetic_path):
                print('Deleting previous synthetic data folder...')
                shutil.rmtree(synthetic_path)
            self.graph_gen_utils = SyntheticDataset(n_nodes=self.n_nodes,
                                                    hidden_dim=self.hidden_dim,
                                                    root=synthetic_path)
        else:
            raise ValueError('The dataset you selected ({}) is not available!'.format(dataset))

    def setup(self):
        print('Building GNN models...')
        # Get the generator depending on the string passed by user
        if self.chosen_model_gen == 'base':
            self.generator = GeneratorNet(n_nodes=7,
                                          hidden_dim=10,
                                          n_ops=self.num_node_features)
        else:
            raise ValueError('The generator you selected ({}) is not available!'.format(self.chosen_model_gen))
        # Move discriminator to GPU or CPU
        self.generator = self.generator.to(self.device)

        print('Setting up training parameters...')
        # Get the discriminator depending on the string passed by user
        if self.chosen_model_dis == 'base':
            self.discriminator = DiscriminatorNet(num_node_features=self.num_node_features,
                                                  num_hidden_features=32)
        else:
            raise ValueError('The discriminator you selected ({}) is not available!'.format(self.chosen_model_dis))
        # Move discriminator to GPU or CPU
        self.discriminator = self.discriminator.to(self.device)

        # Get the optimizer depending on the selected one
        if self.chosen_optimizer == 'sgd':
            self.optimizer_gen = torch.optim.SGD(self.generator.parameters(),
                                                 lr=self.learning_rate,
                                                 momentum=self.momentum,
                                                 weight_decay=self.weight_decay)
            self.optimizer_dis = torch.optim.SGD(self.discriminator.parameters(),
                                                 lr=self.learning_rate,
                                                 momentum=self.momentum,
                                                 weight_decay=self.weight_decay)
        elif self.chosen_optimizer == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.generator.parameters(),
                                                  lr=self.learning_rate,
                                                  weight_decay=self.weight_decay)
            self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(),
                                                  lr=self.learning_rate,
                                                  weight_decay=self.weight_decay)
        else:
            raise ValueError('The optimizer you selected ({}) is not available!'.format(self.chosen_optimizer))

        # Store loss to be used for generation training
        if self.chosen_loss == 'crossentropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('The loss function you selected ({}) is not available!'.format(self.chosen_loss))

        # Setup learning rate scheduler if it is not None
        self.lr_schedulers = []
        if self.chosen_scheduler == 'step':
            self.lr_schedulers.append(optim.lr_scheduler.StepLR(self.optimizer_gen,
                                                                step_size=self.lr_step_size,
                                                                gamma=self.lr_decay))
            self.lr_schedulers.append(optim.lr_scheduler.StepLR(self.optimizer_dis,
                                                                step_size=self.lr_step_size,
                                                                gamma=self.lr_decay))
        elif self.chosen_scheduler == 'plateau':
            if self.lr_metric_to_check == 'loss':
                self.lr_schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen,
                                                                               mode='min',
                                                                               patience=self.lr_step_size,
                                                                               factor=self.lr_decay))
                self.lr_schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_dis,
                                                                               mode='min',
                                                                               patience=self.lr_step_size,
                                                                               factor=self.lr_decay))
            else:
                self.lr_schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen,
                                                                               mode='max',
                                                                               patience=self.lr_step_size,
                                                                               factor=self.lr_decay))
                self.lr_schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_dis,
                                                                               mode='max',
                                                                               patience=self.lr_step_size,
                                                                               factor=self.lr_decay))
        elif self.chosen_scheduler is None:
            self.lr_schedulers = []
        else:
            raise ValueError(
                'The scheduler {} is not available in our trainer implementation!'.format(self.chosen_scheduler))

        print('Setting up metrics...')
        # Setup metrics depending on the choice
        self.metrics = {}
        self.discriminator_metrics = []
        self.generator_metrics = []  # ['val', 'nov'] + ['top_{}'.format(i) for i in range(101)]
        if self.chosen_dataset == 'nas101':
            bench_folder = '{}/NAS101'.format(self.bench_dataset_folder)
        elif self.chosen_dataset == 'nats':
            bench_folder = '{}/NATS'.format(self.bench_dataset_folder)
        else:
            raise ValueError('Dataset {} not available!'.format(self.chosen_dataset))
        for metric in self.chosen_metrics:
            if metric == 'acc':
                self.discriminator_metrics.append(metric)
                self.metrics[metric] = Accuracy()
            elif metric == 'val':
                self.generator_metrics.append(metric)
                self.metrics[metric] = Validity(dataset=self.train_dataset)
            elif metric == 'nov':
                self.generator_metrics.append(metric)
                self.metrics[metric] = Novelty(dataset=self.train_dataset)
            elif metric[:3] == 'top':
                self.generator_metrics.append(metric)
                self.metrics[metric] = InTopKPercent(dataset=self.train_dataset,
                                                     top_k=int(metric.split('_')[-1]))
            else:
                raise ValueError('The metric {} is not available in our implementation yet!'.format(metric))
        # Setup loggers to handle logging
        print('Setting up loggers...')
        self.loggers = {'train_epochs': Logger('log', 'train_epochs', 'train_epochs.log'),
                        'train_steps': Logger('log', 'train_steps', 'train_steps.log'),
                        'models_parameters': Logger('log', 'models_parameters', 'models_parameters.log'), }
        # Log parameters of model
        self.log_models_parameters(epoch=0)

    def save_best_model(self):
        # Check if directory for trained models exists, if not make it
        if not os.path.exists(self.trained_models_folder):
            os.makedirs(self.trained_models_folder)
        # Save generator
        model_name = 'gen_{}_{}_best.pt'.format(self.chosen_model_gen, self.chosen_dataset)
        model_path = os.path.join(self.trained_models_folder, model_name)
        torch.save(self.generator.cpu(), model_path)
        # Save discriminator
        model_name = 'dis_{}_{}_best.pt'.format(self.chosen_model_dis, self.chosen_dataset)
        model_path = os.path.join(self.trained_models_folder, model_name)
        torch.save(self.discriminator.cpu(), model_path)
        # Move models back to gpu if it is used
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def print_models_parameters(self):
        print('Number of parameters of generator: {}'.format(
            sum(p.numel() for p in self.generator.parameters() if p.requires_grad)))
        print('Number of parameters of discriminator: {}'.format(
            sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)))
        for name, param in self.generator.named_parameters():
            if param.requires_grad:
                print('Parameter: {} -> Value: {}'.format(name, param.data))

    def log_models_parameters(self, epoch):
        logger = self.loggers['models_parameters']
        logger.log('{}EPOCH {}\n'.format('\n\n' if epoch == 0 else '', epoch))
        for name, param in self.generator.named_parameters():
            if param.requires_grad:
                logger.log('Parameter: {} -> Value: {}'.format(name, param.data))

    def run(self, print_examples=True):
        print('Start training...')
        # Print an example of randomly generated graph before training
        if print_examples:
            self.print_generated_graph(epoch=0)
        # Define best metric to check in order to store best generator and discriminator
        if self.metric_to_check == 'loss':
            best_met = np.inf
        else:
            best_met = 0.0
        # Iterate over the number of epochs defined in the init
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            print()
            # Log training epoch metrics
            self.log_train_epochs(epoch, train_loss, train_metrics)
            # Save best generator if metric improves
            if self.metric_to_check == 'loss':
                tot_train_loss = train_loss['dis'] + train_loss['gen']
                if tot_train_loss < best_met:
                    best_met = tot_train_loss
                    self.save_best_model()
            elif self.metric_to_check == 'acc':
                if train_metrics[self.metric_to_check] < best_met:
                    best_met = train_metrics[self.metric_to_check]
                    self.save_best_model()
            else:
                if train_metrics[self.metric_to_check] > best_met:
                    best_met = train_metrics[self.metric_to_check]
                    self.save_best_model()
            # Update learning rate depending on the scheduler
            if self.chosen_scheduler == 'step':
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step()
            elif self.chosen_scheduler == 'plateau':
                if self.lr_metric_to_check == 'loss':
                    tot_train_loss = train_loss['dis'] + train_loss['gen']
                    for lr_scheduler in self.lr_schedulers:
                        lr_scheduler.step(tot_train_loss)
                else:
                    for lr_scheduler in self.lr_schedulers:
                        lr_scheduler.step(train_metrics[self.lr_metric_to_check])
            # If requested print one example of validation set predictions
            if print_examples:
                self.print_generated_graph(epoch)
            # Log generator parameters to check if they changed
            self.log_models_parameters(epoch=epoch)
        print('Finished Training')
        # Plot how loss and metrics behaved during training
        self.plot_epochs_summary()
        self.plot_steps_summary()

    def print_generated_graph(self, epoch):
        self.graph_gen_utils.process(batch_size=self.batch_size)
        random_fc_graphs = iter(self.graph_gen_utils.get_data_loader(batch_size=self.batch_size)).next()
        # random_fc_graphs = iter(self.graph_utils.generate_fc_graphs(batch_size=batch_size)).next()
        # x = random_fc_graphs.x
        # edge_index = random_fc_graphs.edge_index
        # batch = random_fc_graphs.batch
        # Move random graphs to device
        random_fc_graphs = random_fc_graphs.to(
            torch.device('cuda' if next(self.generator.parameters()).is_cuda else 'cpu'))
        generated_graph = self.generator(random_fc_graphs)
        generated_graph = generated_graph.cpu()
        # x, edge_index, batch = self.generator(x, edge_index, batch)
        # generated_graph = tg.data.Data(x=x, edge_index=edge_index, batch=batch)
        subgraph_nodes_indices = (generated_graph.batch == 0).nonzero(as_tuple=True)[0]
        # print('Subgraph nodes indices: {}'.format(subgraph_nodes_indices))
        # Get the subgraph containing the first NN graph
        subgraph_nodes = generated_graph.x[subgraph_nodes_indices]
        graph_edges_src = generated_graph.edge_index[0, :]
        graph_edges_dst = generated_graph.edge_index[1, :]
        src_indices_to_keep = [index for index, value in enumerate(graph_edges_src) if value in subgraph_nodes_indices]
        dst_indices_to_keep = [index for index, value in enumerate(graph_edges_dst) if value in subgraph_nodes_indices]
        edges_indices_to_keep = src_indices_to_keep + list(set(dst_indices_to_keep) - set(src_indices_to_keep))
        # print('Subgraph nodes: {}'.format(subgraph_nodes))
        # print('src_indices_to_keep: {}'.format(src_indices_to_keep))
        # print('dst_indices_to_keep: {}'.format(dst_indices_to_keep))
        subgraph_edges = generated_graph.edge_index[:, edges_indices_to_keep]
        # print('subgraph_edges: {}'.format(subgraph_edges))
        subgraph = tg.data.Data(x=subgraph_nodes,
                                edge_index=subgraph_edges)
        # print('subgraph: {}'.format(subgraph))
        # Convert the graph to a networkx graph
        nx_graph = convert_tg_to_nx(subgraph)  # generated_graph)
        nodes_ops_dict = nx.get_node_attributes(nx_graph, 'ops')
        class_name_dict = Utils().get_class_name_dict()
        nodes_labels = {node: class_name_dict[nodes_ops_dict[node]] for node in nx_graph.nodes}
        # nodes_labels = {node: node for node in nx_graph.nodes}
        color_map = Utils().get_color_map(bench='101')
        nodes_colors_dict = {node: color_map[class_name_dict[nodes_ops_dict[node]]] for node in nx_graph.nodes}
        nodes_colors = list(nodes_colors_dict.values())
        nodes_pos = {node: [index, 0] for index, node in enumerate(nx_graph.nodes)}
        # print('Nodes labels: {}'.format(nodes_labels))
        # print('Nodes colors: {}'.format(nodes_colors))
        # print('Nodes positions: {}'.format(nodes_pos))

        plt.rcParams["figure.figsize"] = (12, 7)
        plt.figure()
        nx.draw_networkx_nodes(nx_graph, nodes_pos, node_color=nodes_colors, node_shape='8', node_size=750, alpha=1)
        nx.draw_networkx_labels(nx_graph, nodes_pos, nodes_labels)
        nx.draw_networkx_edges(nx_graph, nodes_pos, arrowstyle="-|>", arrowsize=20, alpha=0.5,
                               connectionstyle="arc3,rad=-0.3")
        plt.axis('off')
        # Save generated graph image
        if not os.path.exists(os.path.join(self.out_path, 'generated_graphs')):
            os.makedirs(os.path.join(self.out_path, 'generated_graphs'))
        image_name = 'epoch_{}.pdf'.format(epoch)
        image_path = os.path.join(self.out_path, 'generated_graphs', image_name)
        plt.savefig(image_path)
        # plt.show()
        plt.close()

    def train_epoch(self, epoch):
        # Set the generator and discriminator to be trainable
        self.generator.train()
        self.discriminator.train()
        avg_loss, avg_metrics = self._train_epoch(epoch)
        return avg_loss, avg_metrics

    def _train_epoch(self, epoch):
        running_loss = {'gen': 0.0, 'dis': 0.0}
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        for batch_index, data in enumerate(self.train_loader, 0):
            batch_loss, batch_scores = self.train_step(data)
            for name, value in batch_loss.items():
                running_loss[name] += value
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = {loss_name: loss_value / (batch_index + 1) for loss_name, loss_value in running_loss.items()}
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            if len(self.train_loader) == 1:
                index_train_batch = 1
            else:
                index_train_batch = batch_index
            self.print_message(epoch,
                               index_train_batch=index_train_batch,
                               train_loss=avg_loss,
                               train_mets=avg_metrics)
            self.log_train_steps(epoch,
                                 train_loss=avg_loss,
                                 train_mets=avg_metrics)
        return avg_loss, avg_metrics

    # @timeit
    def train_step(self, data):
        loss = {'gen': 0.0, 'dis': 0.0}
        data = data.to(self.device)
        # zero the parameter gradients for the discriminator
        self.optimizer_dis.zero_grad()
        # Define labels for true graphs
        batch_size = torch.max(data.batch).item() + 1
        labels_real_graphs = torch.full((batch_size,), 1, device=self.device)
        # forward + backward + optimize
        pred_over_real_graphs = self.discriminator(data)
        # Check if train mask is available
        try:
            loss_d_real = self.criterion(pred_over_real_graphs[data.train_mask],
                                         labels_real_graphs[data.train_mask])
        except AttributeError:
            loss_d_real = self.criterion(pred_over_real_graphs,
                                         labels_real_graphs)
        loss_d_real.backward()
        loss['dis'] += loss_d_real
        # Compute performance metrics
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            if metric_name in self.discriminator_metrics:
                scores[metric_name] = metric_object.compute(pred_over_real_graphs,
                                                            labels_real_graphs)

        # Define labels for generated graphs
        batch_size = torch.max(data.batch).item() + 1
        labels_fake_graphs = torch.full((batch_size,), 0, device=self.device)
        # forward + backward + optimize
        self.graph_gen_utils.process(batch_size=batch_size)
        random_fc_graphs = iter(self.graph_gen_utils.get_data_loader(batch_size=batch_size)).next()
        # Move random graphs to device
        random_fc_graphs = random_fc_graphs.to(self.device)
        fake_graphs = self.generator(random_fc_graphs)
        pred_over_fake_graphs = self.discriminator(fake_graphs)
        # Compute loss over fakes
        loss_d_fake = self.criterion(pred_over_fake_graphs,
                                     labels_fake_graphs)
        loss_d_fake.backward()
        loss['dis'] += loss_d_fake
        # Update discriminator using the optimizer
        self.optimizer_dis.step()
        # Update performance metrics
        for metric_name, metric_object in self.metrics.items():
            if metric_name in self.discriminator_metrics:
                scores[metric_name] += metric_object.compute(pred_over_fake_graphs,
                                                             labels_fake_graphs)
                scores[metric_name] /= 2
            else:
                scores[metric_name] = metric_object.compute(fake_graphs)

        # zero the parameter gradients for the generator
        self.optimizer_gen.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        self.graph_gen_utils.process(batch_size=batch_size)
        random_fc_graphs = iter(self.graph_gen_utils.get_data_loader(batch_size=batch_size)).next()
        # Move random graphs to device
        random_fc_graphs = random_fc_graphs.to(self.device)
        fake_graphs = self.generator(random_fc_graphs)
        pred_over_fake_graphs = self.discriminator(fake_graphs)
        # Calculate G's loss based on this output
        loss_g = self.criterion(pred_over_fake_graphs,
                                labels_real_graphs)
        # Calculate gradients for G
        loss_g.backward()
        loss['gen'] = loss_g
        # Update G
        self.optimizer_gen.step()
        # Get validity metric if requested
        for metric_name, metric_object in self.metrics.items():
            if metric_name in self.generator_metrics:
                scores[metric_name] = metric_object.compute(fake_graphs)
        return loss, scores

    def print_message(self, epoch, index_train_batch, train_loss, train_mets):
        message = '| Epoch: {}/{} |'.format(epoch, self.epochs)
        bar_length = 10
        total_train_batches = len(self.train_loader)
        progress = float(index_train_batch) / float(total_train_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| TRAIN: '
        if train_loss is not None:
            train_loss_message = ''
            for loss_name, loss_value in train_loss.items():
                train_loss_message += '{}_loss={:.5f} '.format(loss_name.capitalize(),
                                                               loss_value)
            message += train_loss_message
        if train_mets is not None:
            train_metrics_message = ''
            for metric_name, metric_value in train_mets.items():
                train_metrics_message += '{}={:.5f} '.format(metric_name,
                                                             metric_value)
            message += train_metrics_message
        message += '|'
        print(message, end='\r')

    def log_train_steps(self, epoch, train_loss, train_mets):
        message = define_train_message_from_epoch_loss_metrics(epoch, train_loss, train_mets)
        logger = self.loggers['train_steps']
        logger.log(message)

    def log_train_epochs(self, epoch, train_loss, train_mets):
        message = define_train_message_from_epoch_loss_metrics(epoch, train_loss, train_mets)
        logger = self.loggers['train_epochs']
        logger.log(message)

    def plot_epochs_summary(self):
        epochs = list(self.epochs_train_summary_dict.keys())
        # print('epochs: {}'.format(epochs))
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Loss and metrics over training and validation epochs')
        loss_and_metrics_keys = list(self.epochs_train_summary_dict[0].keys())
        reshaped_dict_train = {key: [] for key in loss_and_metrics_keys}
        reshaped_dict_val = {key: [] for key in loss_and_metrics_keys}
        for epoch in epochs:
            for key in loss_and_metrics_keys:
                reshaped_dict_train[key].append(self.epochs_train_summary_dict[epoch][key])
                reshaped_dict_val[key].append(self.epochs_val_summary_dict[epoch][key])
        # print('reshaped_dict_train: {}'.format(reshaped_dict_train))
        color_map = plt.cm.get_cmap('hsv', len(loss_and_metrics_keys))
        print('color_map: {}'.format(color_map))
        for index, key in enumerate(loss_and_metrics_keys):
            if 'loss' in key:
                axs[0, 0].plot(reshaped_dict_train[key], label=key, c=color_map(index))
                axs[1, 0].plot(reshaped_dict_val[key], label=key, c=color_map(index))
            else:
                axs[0, 1].plot(reshaped_dict_train[key], label=key, c=color_map(index))
                axs[1, 1].plot(reshaped_dict_val[key], label=key, c=color_map(index))
        axs[0, 0].set_title('Train loss over epochs')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend(loc='upper right')
        axs[0, 1].set_title('Train metrics over epochs')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Metrics')
        axs[0, 1].legend(loc='upper right')
        axs[1, 0].set_title('Validation loss over epochs')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend(loc='upper right')
        axs[1, 1].set_title('Validation metrics over epochs')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Metrics')
        axs[1, 1].legend(loc='upper right')
        plt.tight_layout()
        # Save generated plots image
        if not os.path.exists(os.path.join(self.out_path, 'training_summary')):
            os.makedirs(os.path.join(self.out_path, 'training_summary'))
        image_name = 'epochs.pdf'
        image_path = os.path.join(self.out_path, 'training_summary', image_name)
        plt.savefig(image_path)
        plt.show()

    def plot_steps_summary(self):
        steps_train = list(self.steps_train_summary_dict.keys())
        steps_train.remove('last_step_index')
        # print('steps: {}'.format(steps_train))
        steps_val = list(self.steps_val_summary_dict.keys())
        steps_val.remove('last_step_index')
        # print('steps_val: {}'.format(steps_val))
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Loss and metrics over training and validation steps')
        loss_and_metrics_keys = list(self.steps_train_summary_dict[0].keys())
        reshaped_dict_train = {key: [] for key in loss_and_metrics_keys}
        reshaped_dict_val = {key: [] for key in loss_and_metrics_keys}
        for step in steps_train:
            for key in loss_and_metrics_keys:
                reshaped_dict_train[key].append(self.steps_train_summary_dict[step][key])
        for step in steps_val:
            for key in loss_and_metrics_keys:
                reshaped_dict_val[key].append(self.steps_val_summary_dict[step][key])
        # print('reshaped_dict_train: {}'.format(reshaped_dict_train))
        color_map = plt.cm.get_cmap('hsv', len(loss_and_metrics_keys))
        for index, key in enumerate(loss_and_metrics_keys):
            if 'loss' in key:
                axs[0, 0].plot(reshaped_dict_train[key], label=key, c=color_map(index))
                axs[1, 0].plot(reshaped_dict_val[key], label=key, c=color_map(index))
            else:
                axs[0, 1].plot(reshaped_dict_train[key], label=key, c=color_map(index))
                axs[1, 1].plot(reshaped_dict_val[key], label=key, c=color_map(index))
        axs[0, 0].set_title('Train loss over steps')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend(loc='upper right')
        axs[0, 1].set_title('Train metrics over steps')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].set_ylabel('Metrics')
        axs[0, 1].legend(loc='upper right')
        axs[1, 0].set_title('Validation loss over steps')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend(loc='upper right')
        axs[1, 1].set_title('Validation metrics over steps')
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].set_ylabel('Metrics')
        axs[1, 1].legend(loc='upper right')
        plt.tight_layout()
        # Save generated plots image
        if not os.path.exists(os.path.join(self.out_path, 'training_summary')):
            os.makedirs(os.path.join(self.out_path, 'training_summary'))
        image_name = 'steps.pdf'
        image_path = os.path.join(self.out_path, 'training_summary', image_name)
        plt.savefig(image_path)
        plt.show()


def define_train_message_from_epoch_loss_metrics(epoch, train_loss, train_mets):
    message = 'Epoch: {}, '.format(epoch)
    if train_loss is not None:
        train_loss_message = ''
        for loss_name, loss_value in train_loss.items():
            train_loss_message += '{}_loss: {:.5f}, '.format(loss_name.capitalize(),
                                                            loss_value)
        message += train_loss_message
    if train_mets is not None:
        train_metrics_message = ''
        for metric_name, metric_value in train_mets.items():
            train_metrics_message += '{}: {:.5f}, '.format(metric_name,
                                                          metric_value)
        message += train_metrics_message
    message += ''
    return message


def convert_tg_to_nx(data, previous_graph_index=0):
    graph = nx.DiGraph()
    for index, node in enumerate(data.x):
        if not isinstance(previous_graph_index, int):
            previous_graph_index = previous_graph_index.item()
        graph.add_nodes_from([(index + previous_graph_index, {'ops': torch.argmax(node, dim=-1).item()})])
    for index in range(data.edge_index.shape[1]):
        src = data.edge_index[0, index]
        dst = data.edge_index[1, index]
        graph.add_edge(src.item(), dst.item())
    return graph
