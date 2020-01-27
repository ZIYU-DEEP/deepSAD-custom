# Deep SAD: A Method for Deep Semi-Supervised Anomaly Detection (Custom)
This is repository modifying some part of the original Deep SAD [code](https://github.com/lukasruff/Deep-SAD-PyTorch). The work is ongoing.

Currently, major modifications include:
1. Adding the function (adding some classes in base, datasets, net, main and so on) to support custom datasets.
2. Adding a LSTM autoencoder to support learning of multivariate time series. (Currently you should change the dimensions in it.)
3. Adding a `main_evaluation.py` which provides a more flexible evaluation for the model. Basically, it loads a model and evaluates it on any data you choose.

By the way, based on our data, it seems that the Deep SAD model performs much better without the pretraining phase, which may weaken the argument in the original paper on the upper bound for the entropy of the latent distribution. In other words, it seems that there is no need to maximize the mutual information of $I(X, Z)$, given empirical results.


## Citation
You could find a preprint of the Deep Semi-Supervised Anomaly Detection paper on [arXiv](https://arxiv.org/abs/1906.02694).

```
@article{ruff2019,
  title     = {Deep Semi-Supervised Anomaly Detection},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius},
  journal   = {arXiv preprint arXiv:1906.02694},
  year      = {2019}
}
```

