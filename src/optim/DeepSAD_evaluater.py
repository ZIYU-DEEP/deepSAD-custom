from base.base_trainer_eval import BaseTrainer_eval
from base.base_dataset_eval import BaseADDataset_eval
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

class DeepSADEvaluater(BaseTrainer_eval):

    def __init__(self,
                 c,
                 eta: float,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 60,
                 lr_milestones: tuple = (),
                 batch_size: int = 128,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones,
                         batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.eps = 1e-6
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def evaluate(self, dataset: BaseADDataset_eval, net: BaseNet):
        logger = logging.getLogger()
        all_loader = dataset.loaders(batch_size=self.batch_size,
                                     num_workers=self.n_jobs_dataloader)
        net = net.to(self.device)

        logger.info('Starting evaluating...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()

        with torch.no_grad():
            for data in all_loader:
                inputs, labels, semi_targets, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0,
                                     dist,
                                     self.eta * ((dist + self.eps) **
                                                 (- semi_targets).float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
