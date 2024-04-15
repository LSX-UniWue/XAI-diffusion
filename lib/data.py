from dataclasses import dataclass
from typing import Any, Optional, Union, Dict, List

import numpy as np
import torch

from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]
    cat_transform: Any
    num_transform: Any
    data_name: str
    means_stds: Any

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        if self.data_name == 'erp':
            cat_sizes = [2 for _ in range(24)] + [17, 4, 4, 3, 4, 14, 5, 4, 30, 5, 17, 4, 13, 4, 4, 7, 8, 7] \
                        + [6 for _ in range(10)]  # for num preprocessing = buckets
            # + [2 for i in range(10)]
        elif self.data_name == 'cidds':
            if self.X_num is None:  # means that num_encoding is quantized
                cat_sizes = [2, 5, 4, 3, 36, 19, 4, 36, 19, 4, 4, 5, 2, 2, 2, 2, 2, 2]
            else:
                cat_sizes = [2, 2, 2, 2, 36, 19, 36, 19, 2, 2, 2, 2, 2, 2]
        else:
            cat_sizes = [] if self.X_cat is None else get_category_sizes(self.X_cat[part])
        return cat_sizes


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_fast_dataloader(
        D: Dataset,
        split: str,
        batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split == 'train'))
    while True:
        yield from dataloader
