from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod
import numpy as np
import torch

from utils import torch_cos_sim, torch_unique, expand


class AbstractManager(ABC):
    @abstractmethod
    def splits(self, x, n_parts):
        pass

    @abstractmethod
    def cos_similarity(self, x, y):
        pass

    @abstractmethod
    def unique(self, x):
        pass

    @abstractmethod
    def where(self, x):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, x, path):
        pass

    @abstractmethod
    def prepare_updates(self, origs, sims, scores):
        pass


class NumpyManager(AbstractManager):
    def splits(self, x, n_parts):
        return np.array_split(x, n_parts)

    def cos_similarity(self, x, y):
        return cosine_similarity(x, y)

    def unique(self, x):
        return np.unique(x, return_index=True)

    def where(self, x):
        return np.where(x)

    def load(self, path):
        return np.load(path)

    def save(self, x, path):
        return np.save(x, path)

    def prepare_updates(self, origs, sims, scores):
        return origs, sims, scores


class TorchManager(AbstractManager):
    def splits(self, x, n_parts):
        split_size = x.shape[0] // n_parts
        return torch.split(x, split_size)

    def cos_similarity(self, x, y):
        return torch_cos_sim(x, y)

    def unique(self, x):
        return torch_unique(x.cpu())

    def where(self, x):
        return torch.where(x.cpu())

    def load(self, path):
        return torch.load(path)

    def save(self, x, path):
        return torch.save(x, path)

    def prepare_updates(self, origs, sims, scores):
        return expand(origs), expand(sims), expand(scores)
