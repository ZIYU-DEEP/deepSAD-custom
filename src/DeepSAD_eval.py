import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_evaluater import DeepSADEvaluater


class DeepSAD_eval(object):
    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c
        self.net_name = None
        self.net = None  # neural network phi
        self.trainer = None
        self.optimizer_name = None
        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None
        self.results = {'train_time': None, 'test_auc': None, 'test_time': None, 'test_scores': None}
        self.ae_results = {'train_time': None, 'test_auc': None, 'test_time': None}

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def evaluate(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADEvaluater(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.evaluate(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores


    def load_model(self, model_path, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as f:
            json.dump(self.results, f)
