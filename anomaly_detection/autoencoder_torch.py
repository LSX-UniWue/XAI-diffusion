import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from tab_ddpm import index_to_onehot


class Autoencoder(torch.nn.Module):
    """
    Autoencoder being build with n_bottleneck neurons in bottleneck.
    Encoder and decoder contain n_layers each.
    size of layers starts at 2**(log2(n_bottleneck) + 1) near bottleneck and increases with 2**(last+1)
    """

    def __init__(self, n_inputs, cpus=0, n_layers=3, n_bottleneck=2 ** 3, batch_norm=False, seed=0, **params):
        # setting number of threads for parallelization
        super(Autoencoder, self).__init__()

        torch.manual_seed(seed)
        if cpus > 0:
            torch.set_num_threads(cpus * 2)

        bottleneck_exp = (np.log2(n_bottleneck))

        # AE architecture
        layers = []
        # Input
        layers.append(torch.nn.Linear(in_features=n_inputs,
                                      out_features=int(2 ** (bottleneck_exp + n_layers))))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(num_features=int(2 ** (bottleneck_exp + n_layers))))
        layers.append(torch.nn.ReLU())

        # Encoder
        for i in range(n_layers - 1, 0, -1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers.append(torch.nn.Linear(in_features=int(2 ** (bottleneck_exp + i + 1)),
                                          out_features=int(2 ** (bottleneck_exp + i))))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(num_features=int(2 ** (bottleneck_exp + i))))
            layers.append(torch.nn.ReLU())

        # Bottleneck
        layers += [torch.nn.Linear(in_features=int(2 ** (bottleneck_exp + 1)),
                                   out_features=n_bottleneck)]
        # Decoder
        for i in range(1, n_layers + 1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers += [torch.nn.Linear(in_features=int(2 ** (bottleneck_exp + i - 1)),
                                       out_features=int(2 ** (bottleneck_exp + i))),
                       torch.nn.ReLU()]
        # Output
        layers += [torch.nn.Linear(in_features=int(2 ** (bottleneck_exp + n_layers)),
                                   out_features=n_inputs)]  # output layer
        # Full model
        self.model = torch.nn.Sequential(*layers)
        self.add_module('model', self.model)
        self.add_module('distance_layer', module=torch.nn.PairwiseDistance(p=2))

        if 'learning_rate' in params:
            self.optim = torch.optim.Adam(params=self.model.parameters(), lr=params.pop('learning_rate'))
        else:
            self.optim = torch.optim.Adam(params=self.model.parameters())

        self.params = params

    def build_embedding_model(self):
        """copies the Sequential model but cuts off after the smallest latent embedding layer"""
        self.embedd_model = torch.nn.Sequential(*list(self.model.children())[:len(self.model._modules) // 2])

    # def build_norm_output_model(self):
    #     """copies the Sequential model but cuts off to give intermediate outputs after each batch norm layer"""

    def score_samples(self, x, output_to_numpy=True, invert_score=True, is_ordinal=False, cat_encoder=None,
                      cat_encoder_ddpm=None, data_name=None, device='cpu'):
        x = self.to_tensor(x)
        # convert x to ohe if it is ordinal
        if is_ordinal:
            if data_name == 'erp':
                n_cat_feat = cat_encoder_ddpm.encoder.n_features_in_

                x_cat = x[:, :n_cat_feat]
                x_num = x[:, n_cat_feat:]

                x_cat_raw = cat_encoder_ddpm.inverse_transform(x_cat.cpu())
                cols = cat_encoder.encoder.cols
                x_cat_raw_df = pd.DataFrame(x_cat_raw, columns=cols)

                x_cat_ohe = cat_encoder.transform(x_cat_raw_df)
                x_cat_ohe = self.to_tensor(x_cat_ohe)

                num_classes = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                x_num_ohe = index_to_onehot(self.to_tensor(x_num).long(), num_classes)
                x = torch.cat([x_cat_ohe.to('cpu'), x_num_ohe.to('cpu')], dim=1)
            elif data_name == 'cidds':
                x = index_to_onehot(x.long(), [2, 5, 4, 3, 36, 19, 4, 36, 19, 4, 4, 5, 2, 2, 2, 2, 2, 2])
            else:
                raise ValueError

        if data_name == 'cidds':
            # retransform complete ohe to partial ohe to match AE input dimensions
            # drop the 'attribute0' columns and keep 'attribute1' columns, same as transforming binary ohe to ordinal
            indices = [0, 141, 143, 145, 147, 149, 151]
            mask = torch.ones(x.size(1), dtype=torch.bool)
            mask[indices] = False
            x = x[:, mask].float()

        x = x.to(device)
        self.to(device)
        loss = self.__call__(input_tensor=x)
        if invert_score:
            loss = -1 * loss
        if output_to_numpy:
            return loss.data.cpu().numpy()
        else:
            return loss

    def to_tensor(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        return x

    def embedd(self, x, output_to_numpy=True):
        x = self.to_tensor(x)
        assert hasattr(self, 'embedd_model'), 'Embedding model must be initialized by calling build_embedding_model()'
        out = self.embedd_model(x)
        if output_to_numpy:
            return out.data.numpy()
        else:
            return out

    def reconstruct(self, x, output_to_numpy=True):
        x = self.to_tensor(x)
        out = self.model(x)
        if output_to_numpy:
            return out.data.numpy()
        else:
            return out

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path, only_model=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if only_model:
            self.model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            self.load_state_dict(torch.load(load_path, map_location=device))
        return self

    def __call__(self, input_tensor, return_pre_norm_logits=False, *args, **kwargs):

        if return_pre_norm_logits:
            batch_norm_outputs = []
            x = input_tensor
            for layer in self.model:
                x = layer(x)
                if isinstance(layer, torch.nn.BatchNorm1d):
                    batch_norm_outputs.append({'logits': x, 'mean': layer.running_mean, 'var': layer.running_var})
            anom_score = self.distance_layer(x, input_tensor)
            return anom_score, batch_norm_outputs  # the higher, the more abnormal (reconstruction error)

        else:
            pred = self.model(input_tensor)
            anom_score = self.distance_layer(pred, input_tensor)
            return anom_score  # the higher, the more abnormal (reconstruction error)

    def fit(self, data, device='cpu'):
        verbose = self.params['verbose']
        if isinstance(data, torch.utils.data.DataLoader):
            data_loader = data
        else:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(data.values), torch.zeros((data.values.shape[0], 1)))
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.params['batch_size'],
                                                      shuffle=True,
                                                      num_workers=0)
        self.to(device)
        for _ in tqdm(range(self.params['epochs']), disable=verbose < 1):
            counter = 0
            for x, _ in tqdm(data_loader, disable=verbose < 2):
                x = x.to(device)
                y_pred = self.model(x)
                loss = self.distance_layer(y_pred, x).mean()

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                counter += 1

        return self

    def test(self, data, device='cpu', return_metrics=True):
        verbose = self.params['verbose']

        if isinstance(data, torch.utils.data.DataLoader):
            data_loader = data
        else:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(data.values), torch.Tensor(data.values))
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.params['batch_size'],
                                                      shuffle=False,
                                                      num_workers=0)

        preds = []
        trues = []
        self.eval()
        self.to(device)

        for x, y in tqdm(data_loader, disable=verbose < 2):
            x = x.to(device)
            anom_score = self(x)
            preds.extend(list(anom_score.detach().cpu().numpy()))
            trues.extend(list(y.detach().cpu().numpy()))
        scores = np.array(preds)
        y = np.array(trues)

        if return_metrics:
            return {f'auc_pr': average_precision_score(y_true=y, y_score=scores),
                    f'auc_roc': roc_auc_score(y_true=y, y_score=scores)}

        else:
            return {'pred': scores, 'true': y}
