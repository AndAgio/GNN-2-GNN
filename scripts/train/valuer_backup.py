import os
import numpy as np
import torch
import torch.optim as optim
# Import my modules
from scripts.data import NASDataset3Splits as NASDataset
from scripts.models import ValueNet
from scripts.metrics import MSE, MAE


class ValuerTrainer():
    def __init__(self, model, dataset, optimizer='SGD', momentum=0.9,
                 weight_decay=5e-4, loss_reg='mse', loss_class='crossentropy',
                 metrics=['mse'], batch_size=None, epochs=100, lr=0.01,
                 lr_sched=None, lr_decay=None, lr_step_size=None,
                 lr_metric_to_check='loss',
                 metric_to_check='acc',
                 dataset_folder='gnn2gnn_datasets',
                 out_path='outputs'):
        self.chosen_model = model
        self.chosen_optimizer = optimizer
        self.batch_size = batch_size if batch_size is None else int(batch_size)
        self.chosen_loss_reg = loss_reg
        self.chosen_loss_class = loss_class
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
        self.dataset_folder = dataset_folder
        self.out_path = os.path.join(out_path, 'regression')
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
                                            split='train')
            self.val_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                          split='val')
            self.test_dataset = NASDataset(root=os.path.join(self.dataset_folder, 'NAS101'),
                                           split='test')
            self.train_loader = self.train_dataset.get_data_loader(batch_size=self.batch_size,
                                                                   shuffle=True)
            self.val_loader = self.val_dataset.get_data_loader(batch_size=self.batch_size,
                                                               shuffle=False)
            self.test_loader = self.test_dataset.get_data_loader(batch_size=self.batch_size,
                                                                 shuffle=False)
            # Get the number of node features and set number of classes
            self.num_node_features = self.train_dataset.num_features
            self.num_classes = self.train_dataset.num_classes
            print('Number of features: {}'.format(self.num_node_features))
            print('Number of classes: {}'.format(self.num_classes))
            print('Train loader: {}'.format(self.train_loader))
        # elif dataset == 'nats':
        #     natsbench = NATSBench(split='topo')
        #     self.dataset = natsbench.get_dataset()
        #     loaders = natsbench.get_dataloaders()
        #     self.train_loader, self.val_loader, self.test_loader = loaders
        #     # Get the number of node features and set number of classes
        #     self.num_node_features = self.dataset.num_node_features
        #     self.num_classes = 1
        else:
            raise ValueError('The dataset you selected ({}) is not available!'.format(dataset))

    def setup(self):
        # Get the valuer depending on the string passed by user
        if self.chosen_model == 'base':
            self.valuer = ValueNet(num_node_features=self.num_node_features,
                                          num_hidden_features=32)
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
        if self.chosen_loss_reg == 'mse':
            self.criterion_reg = torch.nn.MSELoss()
        else:
            raise ValueError(
                'The loss function you selected ({}) is not available for regression!'.format(self.chosen_loss_reg))

        # Store loss to be used for classification training
        if self.chosen_loss_class == 'crossentropy':
            self.criterion_class = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('The loss function you selected ({}) is not available for classification!'.format(
                self.chosen_loss_class))

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
            else:
                raise ValueError('The metric {} is not available in our implementation yet!'.format(metric))
        print('Number of parameters of valuer: {}'.format(
            sum(p.numel() for p in self.valuer.parameters() if p.requires_grad)))

    def save_best_model(self):
        # Check if directory for trained models exists, if not make it
        if not os.path.exists(self.trained_models_folder):
            os.makedirs(self.trained_models_folder)
        model_name = 'reg_{}_{}_best.pt'.format(self.chosen_model, self.chosen_dataset)
        model_path = os.path.join(self.trained_models_folder, model_name)
        torch.save(self.valuer.cpu(), model_path)

    def run(self, print_examples=True):
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
                for _, data in enumerate(self.val_loader, 0):
                    data = data.to(self.device)
                    outs_reg, _ = self.valuer(data)
                    print('Outputs: {}'.format(outs_reg))
                    print('Data labels: {}'.format(data['test_accuracy']))
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
        outs_reg, outs_class = self.valuer(data)
        print('Outputs: {}'.format(outs_reg))
        print('Data labels: {}'.format(data['test_accuracy']))
        print('Data: {}'.format(data))
        # Check if train mask is available
        loss_reg = self.criterion_reg(outs_reg, data['test_accuracy'])
        loss_class = self.criterion_class(outs_class, data['y_class'])
        loss = loss_reg + loss_class
        loss.backward()
        self.optimizer.step()
        # Compute performance metrics
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(outs_reg, data['test_accuracy'])
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
        for batch_index, data in enumerate(self.val_loader, 0):
            batch_loss, batch_scores = self.val_step(data)
            running_loss += batch_loss
            for metric_name, metric_value in batch_scores.items():
                running_scores[metric_name] += metric_value
            avg_loss = running_loss / (batch_index + 1)
            avg_metrics = {met_name: met_value / (batch_index + 1) for met_name, met_value in running_scores.items()}
            if len(self.val_loader) == 1:
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
        outs_reg, outs_class = self.valuer(data)
        loss_reg = self.criterion_reg(outs_reg, data['test_accuracy'])
        loss_class = self.criterion_class(outs_class, data['y_class'])
        loss = loss_reg + loss_class
        # Compute performance metrics
        scores = {}
        for metric_name, metric_object in self.metrics.items():
            scores[metric_name] = metric_object.compute(outs_reg, data['test_accuracy'])
        # return statistics
        return loss.item(), scores

    def print_message(self, epoch, index_train_batch, train_loss, train_mets,
                      index_val_batch, val_loss, val_mets):
        message = '| Epoch: {}/{} |'.format(epoch + 1, self.epochs)
        bar_length = 10
        if self.batch_size is None:
            total_train_batches = len(self.dataset)
        else:
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
            if self.batch_size is None:
                total_val_batches = len(self.dataset)
            else:
                total_val_batches = len(self.val_loader)
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
