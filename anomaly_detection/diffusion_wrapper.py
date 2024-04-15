import torch
import pandas as pd
import dask.array as da
from dask_ml.wrappers import ParallelPostFit

from tab_ddpm import index_to_onehot


class DiffusionADWrapper:
    """Wraps a sklearn detector to be used with Diffusion inpainting."""
    def __init__(self, detector):
        self.detector = detector

    def fit(self, *args, **kwargs):
        self.detector = self.detector.fit(*args, **kwargs)
        return self

    def score_samples(self, X, *args, is_ordinal=False, cat_encoder=None, cat_encoder_ddpm=None,
                      data_name=None, device=None, **kwargs):

        if isinstance(self.detector, ParallelPostFit) and data_name == 'cidds':  # trick for multiprocessing single core algorithms with dask for oc-svm
            if is_ordinal:
                if not torch.is_tensor(X):
                    X = torch.Tensor(X)
                X = index_to_onehot(X.long(), [2, 5, 4, 3, 36, 19, 4, 36, 19, 4, 4, 5, 2, 2, 2, 2, 2, 2])

            # retransform complete ohe to partial ohe to match detector input dimensions
            # drop the 'attribute0' columns and keep 'attribute1' columns, same as transforming binary ohe to ordinal
            indices = [0, 141, 143, 145, 147, 149, 151]
            mask = torch.ones(X.shape[1], dtype=torch.bool)
            mask[indices] = False
            X = X[:, mask]
            if torch.is_tensor(X):
                X = X.detach().cpu().numpy()

        # convert x to ohe if it is ordinal
        if is_ordinal:
            if data_name == 'erp':
                n_cat_feat = cat_encoder_ddpm.encoder.n_features_in_

                x_cat = X[:, :n_cat_feat]
                x_num = X[:, n_cat_feat:]

                if torch.is_tensor(x_cat):
                    x_cat = x_cat.cpu().clone()
                else:
                    x_cat = x_cat.copy()
                x_cat_raw = cat_encoder_ddpm.inverse_transform(x_cat)  # TODO: warning: x_cat changes inplace
                cols = cat_encoder.encoder.cols
                x_cat_raw_df = pd.DataFrame(x_cat_raw, columns=cols)

                x_cat_ohe = cat_encoder.transform(x_cat_raw_df)
                x_cat_ohe = torch.Tensor(x_cat_ohe.values)

                num_classes = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                if torch.is_tensor(x_num):
                    x_num = x_num.long()
                else:
                    if isinstance(x_num, pd.DataFrame):
                        x_num = x_num.values
                    x_num = torch.Tensor(x_num).long()
                x_num_ohe = index_to_onehot(x_num, num_classes)
                X = torch.cat([x_cat_ohe.to('cpu'), x_num_ohe.to('cpu')], dim=1)

            elif data_name == 'cidds':
                if not torch.is_tensor(X):
                    X = torch.Tensor(X)
                X = index_to_onehot(X.long(), [2, 5, 4, 3, 36, 19, 4, 36, 19, 4, 4, 5, 2, 2, 2, 2, 2, 2])
            else:
                raise ValueError

        if data_name == 'cidds':
            # retransform complete ohe to partial ohe to match AE input dimensions
            # drop the 'attribute0' columns and keep 'attribute1' columns, same as transforming binary ohe to ordinal
            indices = [0, 141, 143, 145, 147, 149, 151]
            mask = torch.ones(X.size(1), dtype=torch.bool)
            mask[indices] = False
            X = X[:, mask].float()

        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()

        if isinstance(self.detector, ParallelPostFit):
            data = da.from_array(X, chunks=(100, -1))
            out = self.detector.predict(data).compute()
        else:
            out = self.detector.score_samples(X, *args, **kwargs)
        return out

