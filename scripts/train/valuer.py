import os
import numpy as np
import torch
import torch.optim as optim
# Import my modules
from scripts.data import NASDataset2Splits as NASDataset
from scripts.data import NATSDataset2Splits as NATSDataset
from scripts.models import ValueNet
from scripts.metrics import MSE, MAE, Accuracy, GeneratorMetrics, ModelAccuracy


class ValuerTrainer():
    def __init__(self, model, dataset, optimizer='SGD', momentum=0.9,
                 weight_decay=5e-4, is_reg=False,
                 batch_size=None, epochs=100, lr=0.01,
                 lr_sched=None, lr_decay=None, lr_step_size=None,
                 dataset_folder='gnn2gnn_datasets',
                 bench_dataset_folder='nas_benchmark_datasets',
                 complexity=1,
                 sub_skip=False,
                 nats_data='cifar10',
                 out_path='outputs'):
        self.chosen_model = model
        self.chosen_optimizer = optimizer
        self.batch_size = batch_size if batch_size is None else int(batch_size)
        self.is_reg = is_reg
        self.chosen_loss = 'mse' if self.is_reg else 'crossentropy'
        self.chosen_metrics = ['mse', 'mae'] if self.is_reg else ['acc']
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
                self.lr_metric_to_check = 'mse' if self.is_reg else 'acc'
        self.metric_to_check = 'mse' if self.is_reg else 'acc'
        self.dataset_folder = dataset_folder
        self.sub_skip = False if dataset == 'nas101' else sub_skip
        self.nats_data = nats_data
        self.bench_dataset_folder = bench_dataset_folder
        self.complexity = complexity
        self.out_path = os.path.join(out_path, 'value_predictor')
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get dataset and setup the trainer
        self.get_dataset(dataset)
        self.setup()

    def get_dataset(self, dataset):
        self.chosen_dataset = dataset
        # Get the dataset depending on the selected one and move it to the torch device
        if dataset == 'nas101':
            self.train_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                            bench_folder=os.path.join(self.bench_dataset_folder, 'NAS101'),
                                            split='train', complexity=self.complexity)
            self.test_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                           bench_folder=os.path.join(self.bench_dataset_folder, 'NAS101'),
                                           split='test', complexity=self.complexity)
            print('self.test_dataset[0]: {}'.format(self.test_dataset[0]))
            self.train_loader = self.train_dataset.get_data_loader(batch_size=self.batch_size,
                                                                   shuffle=True)
            self.test_loader = self.test_dataset.get_data_loader(batch_size=self.batch_size,
                                                                 shuffle=False)
            # Get the number of node features and set number of classes
            self.n_nodes = self.train_dataset.get_num_nodes()
            print('self.n_nodes:', self.n_nodes)
            self.num_node_features = self.train_dataset.num_features
            print('self.num_node_features:', self.num_node_features)
            self.num_classes = self.train_dataset.num_classes
            print('self.num_classes:', self.num_classes)
        elif dataset == 'nats':
            print('self.nats_data: {}'.format(self.nats_data))
            self.train_dataset = NATSDataset(root=os.path.join(self.dataset_folder, 'NATS'),
                                             bench_folder=os.path.join(self.bench_dataset_folder, 'NATS'),
                                             split='train',
                                             chosen_data=self.nats_data,
                                             sub_skip=self.sub_skip, complexity=self.complexity)
            self.test_dataset = NATSDataset(root=os.path.join(self.dataset_folder, 'NATS'),
                                            bench_folder=os.path.join(self.bench_dataset_folder, 'NATS'),
                                            split='test',
                                            chosen_data=self.nats_data,
                                            sub_skip=self.sub_skip, complexity=self.complexity)
            print('self.train_dataset[0]: {}'.format(self.train_dataset[0]))
            print('self.train_dataset[0].x: {}'.format(self.train_dataset[0].x))
            self.train_loader = self.train_dataset.get_data_loader(batch_size=self.batch_size,
                                                                   shuffle=True)
            self.test_loader = self.test_dataset.get_data_loader(batch_size=self.batch_size,
                                                                 shuffle=False)
            # Get the number of node features and set number of classes
            self.n_nodes = self.train_dataset.get_num_nodes()
            print('self.n_nodes:', self.n_nodes)
            self.num_node_features = self.train_dataset.num_features
            print('self.num_node_features:', self.num_node_features)
            self.num_classes = self.train_dataset.num_classes
            print('self.num_classes:', self.num_classes)
        else:
            raise ValueError('The dataset you selected ({}) is not available!'.format(dataset))

    def setup(self):
        # Get the valuer depending on the string passed by user
        if self.chosen_model == 'base':
            self.valuer = ValueNet(num_node_features=self.num_node_features,
                                          num_hidden_features=32,
                                          is_reg=self.is_reg)
        else:
            raise ValueError('The valuer you selected ({}) is not available!'.format(self.chosen_model))
        # Move valuer to GPU or CPU
        self.valuer = self.valuer.to(self.device)

        # Get the optimizer depending on the selected one
        if self.chosen_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.valuer.parameters(),
                                             lr=self.learning_rate,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        elif self.chosen_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.valuer.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('The optimizer you selected ({}) is not available!'.format(self.chosen_optimizer))

        # Store loss to be used for regression training
        if self.chosen_loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.chosen_loss == 'crossentropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(
                'The loss function you selected ({}) is not available for regression!'.format(self.chosen_loss))

        # Setup learning rate scheduler if it is not None
        if self.chosen_scheduler == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=self.lr_step_size,
                                                          gamma=self.lr_decay)
        elif self.chosen_scheduler == 'plateau':
            if self.lr_metric_to_check == 'loss':
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                         mode='min',
                                                                         patience=self.lr_step_size,
                                                                         factor=self.lr_decay)
            else:
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                         mode='max',
                                                                         patience=self.lr_step_size,
                                                                         factor=self.lr_decay)
        elif self.chosen_scheduler is None:
            self.lr_scheduler = None
        else:
            raise ValueError(
                'The scheduler {} is not available in our trainer implementation!'.format(self.chosen_scheduler))

        # Setup metrics depending on the choice
        self.metrics = {}
        for metric in self.chosen_metrics:
            if metric == 'mse':
                self.metrics[metric] = MSE()
            elif metric == 'mae':
                self.metrics[metric] = MAE()
            elif metric == 'acc':
                self.metrics[metric] = Accuracy()
            else:
                raise ValueError('The metric {} is not available in our implementation yet!'.format(metric))
        gen_metrics = GeneratorMetrics(self.train_dataset)
        dataset_metrics = gen_metrics.get_dataset_metrics()
        # Append metric for computing model accuracy (used by valuer)
        self.model_accuracy_metric = ModelAccuracy(dataset_metrics=dataset_metrics,
                                                   is_reg=self.is_reg)
        print('Number of parameters of valuer: {}'.format(
            sum(p.numel() for p in self.valuer.parameters() if p.requires_grad)))

    def save_best_model(self):
        # Check if directory for trained models exists, if not make it
        if not os.path.exists(self.trained_models_folder):
            os.makedirs(self.trained_models_folder)
        model_name = 'reg_{}_{}_best.pt'.format(self.chosen_model, self.chosen_dataset)
        model_path = os.path.join(self.trained_models_folder, model_name)
        torch.save(self.valuer.cpu(), model_path)

    def run(self, print_examples=False):
        print('Start training...')
        # Define best metric to check in order to store best valuer
        if self.metric_to_check == 'loss':
            best_met = np.inf
        else:
            best_met = 0.0
        # Iterate over the number of epochs defined in the init
        for epoch in range(self.epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            # Validate
            val_loss, val_metrics = self.val_epoch(epoch, train_loss, train_metrics)
            print()
            # Save best valuer if metric improves
            if self.metric_to_check == 'loss':
                if val_loss < best_met:
                    best_met = val_loss
                    self.save_best_model()
            else:
                if val_metrics[self.metric_to_check] > best_met:
                    best_met = val_metrics[self.metric_to_check]
                    self.save_best_model()
            # Update learning rate depending on the scheduler
            if self.chosen_scheduler == 'step':
                self.lr_scheduler.step()
            elif self.chosen_scheduler == 'plateau':
                if self.lr_metric_to_check == 'loss':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step(val_metrics[self.lr_metric_to_check])
            # If requested print one example of validation set predictions
            if print_examples:
                for _, data in enumerate(self.test_loader, 0):
                    data = data.to(self.device)
                    outs = self.valuer(data)
                    print('Outputs: {}'.format(outs))
                    print('Models test accuracy: {}'.format(data['test_accuracy']))
                    accuracies = self.model_accuracy_metric.compute(data)
                    labels = torch.tensor(accuracies, device=self.device)
                    print('Data labels: {}'.format(labels))
                    break
        print('Finished Training')

    def train_epoch(self, epoch):
        # Set the valuer to be trainable
        self.valuer.train()
        avg_loss, avg_metrics = self._train_epoch(epoch)
        return avg_loss, avg_metrics

    def _train_epoch(self, epoch):
        running_loss = 0.0
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        for batch_index, data in enumerate(self.train_loader, 0):
            batch_loss, batch_scores = self.train_step(data)
            running_loss += batch_loss
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = running_loss / (batch_index + 1)
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            if len(self.train_loader) == 1:
                index_train_batch = 1
            else:
                index_train_batch = batch_index
            self.print_message(epoch,
                               index_train_batch=index_train_batch,
                               train_loss=avg_loss,
                               train_mets=avg_metrics,
                               index_val_batch=None,
                               val_loss=None,
                               val_mets=None)
        return avg_loss, avg_metrics

    def train_step(self, data):
        data = data.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outs = self.valuer(data)
        # print('Outputs: {}'.format(outs))
        accuracies = self.model_accuracy_metric.compute(data)
        labels = torch.tensor(accuracies, device=self.device)
        # print('Data labels: {}'.format(accuracies))
        # Check if train mask is available
        loss = self.criterion(outs, labels)
        loss.backward()
        self.optimizer.step()
        # Compute performance metrics
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(outs, labels)
        # return statistics
        return loss.item(), scores

    @torch.no_grad()
    def val_epoch(self, epoch, train_loss, train_mets):
        # Set the valuer to be non trainable
        self.valuer.eval()
        avg_loss, avg_metrics = self._val_epoch(epoch, train_loss, train_mets)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def _val_epoch(self, epoch, train_loss, train_mets):
        running_loss = 0.0
        running_scores = {met_name: 0.0 for met_name in self.metrics.keys()}
        for batch_index, data in enumerate(self.test_loader, 0):
            batch_loss, batch_scores = self.val_step(data)
            running_loss += batch_loss
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = running_loss / (batch_index + 1)
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            if len(self.test_loader) == 1:
                index_val_batch = 1
            else:
                index_val_batch = batch_index
            self.print_message(epoch,
                               index_train_batch=len(self.train_loader),
                               train_loss=train_loss,
                               train_mets=train_mets,
                               index_val_batch=index_val_batch,
                               val_loss=avg_loss,
                               val_mets=avg_metrics)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def val_step(self, data):
        data = data.to(self.device)
        # Get output and loss
        outs = self.valuer(data)
        accuracies = self.model_accuracy_metric.compute(data)
        labels = torch.tensor(accuracies, device=self.device)
        # Check if train mask is available
        loss = self.criterion(outs, labels)
        # Compute performance metrics
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(outs, labels)
        # return statistics
        return loss.item(), scores

    def print_message(self, epoch, index_train_batch, train_loss, train_mets,
                      index_val_batch, val_loss, val_mets):
        message = '| Epoch: {}/{} |'.format(epoch + 1, self.epochs)
        bar_length = 10
        total_train_batches = len(self.train_loader)
        progress = float(index_train_batch) / float(total_train_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| TRAIN: loss={:.5f} '.format(train_loss)
        if train_mets is not None:
            train_metrics_message = ''
            for metric_name, metric_value in train_mets.items():
                train_metrics_message += '{}={:.5f} '.format(metric_name,
                                                             metric_value)
            message += train_metrics_message
        # Add validation loss
        if val_mets is not None:
            bar_length = 10
            total_val_batches = len(self.test_loader)
            progress = float(index_val_batch) / float(total_val_batches)
            if progress >= 1.:
                progress = 1
            block = int(round(bar_length * progress))
            message += '|[{}]'.format('=' * block + ' ' * (bar_length - block))
            message += '| VAL: loss={:.5f} '.format(val_loss)
            val_metrics_message = ''
            for metric_name, metric_value in val_mets.items():
                val_metrics_message += '{}={:.5f} '.format(metric_name,
                                                           metric_value)
            message += val_metrics_message
        message += '|'
        # message += 'Loss weights are: {}'.format(self.criterion_reg.weight.numpy())
        print(message, end='\r')
