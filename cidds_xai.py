
import os

import functools
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from xai.util import tabular_reference_points
from data.cidds.util import get_cols_and_dtypes, get_column_mapping, get_summed_columns
from anomaly_detection.diffusion_wrapper import DiffusionADWrapper
from data.data_handling import make_dataset_shap
from repaint import conf_mgt
from repaint.guided_diffusion import dist_util
from repaint.utils import yamlread
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model

class DaskOCSVM:
    """Small wrapper to trick dask_ml into parallelizing anomaly detection methods"""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.score_samples(X)


def xai_to_categorical(df, cat_encoding='onehot', num_encoding='quantized'):
    # sum all encoded scores to single categorical values for each column
    categories = get_column_mapping(cat_encoding=cat_encoding, num_encoding=num_encoding, as_int=True)
    category_names = get_summed_columns()

    data = df.values
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    data_cat = pd.DataFrame(data_cat, columns=category_names, index=df.index)
    return data_cat


def get_expl_scores(explanation, gold_standard, score_type='auc_roc', to_categorical=False, cat_encoding='onehot', num_encoding='quantized'):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    # Explanation values for each feature treated as likelihood of anomalous feature
    #  -aggregated to feature-scores over all feature assignments
    #  -flattened to match shape of y_true
    #  -inverted, so higher score means more anomalous
    if to_categorical:
        explanation = xai_to_categorical(explanation, cat_encoding=cat_encoding, num_encoding=num_encoding)
    scores = []
    for i, row in explanation.iterrows():
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.iloc[i], y_score=row))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.iloc[i].values.reshape(1, -1), row.values.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   model,
                   gold_standard_path,
                   expl_path,
                   xai_type,
                   out_path,
                   cat_encoding='onehot',
                   num_encoding='quantized',
                   to_categorical=False):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(expl_path, header=0, index_col=0)

    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    if model in ['IF', 'OCSVM', 'AE']:
        expl = -1 * expl

    if model == 'DDPM':
        # Sum up explanation scores for protocol
        expl['Traffic'] = expl['isICMP'] + expl['isUDP'] + expl['isTCP']
        expl = expl.drop(['isICMP', 'isUDP', 'isTCP'], axis=1)

        # reorder columns
        column_order = ['isWeekday', 'Daytime', 'Duration', 'Traffic', 'Src IP', 'Src Pt', 'Src Conns', 'Dst IP', 'Dst Pt', 'Dst Conns', 'Packets', 'Bytes', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
        expl = expl[column_order]

    # Load gold standard explanations and convert to pd.Series containing
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
    gold_expl = (gold_expl == 'X')

    assert expl.shape[0] == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {expl.shape[0]}"

    roc_mean, roc_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='auc_roc',
                                        to_categorical=to_categorical,
                                        cat_encoding=cat_encoding,
                                        num_encoding=num_encoding)
    cos_mean, cos_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='cosine_sim',
                                        to_categorical=to_categorical,
                                        cat_encoding=cat_encoding,
                                        num_encoding=num_encoding)

    out_dict = {'xai': xai_type,
                'variant': background,
                f'ROC': roc_mean,
                f'ROC-std': roc_std,
                f'Cos': cos_mean,
                f'Cos-std': cos_std}
    [print(key + ':', val) for key, val in out_dict.items()]

    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()

        out_df = pd.concat([out_df, pd.DataFrame([out_dict])], ignore_index=True)

        out_df.to_csv(out_path, index=False)
    return out_dict


def explain_anomalies(compare_with_gold_standard,
                      expl_folder,
                      job_name,
                      conf,
                      xai_type='shap',
                      model='AE',
                      background='zeros',
                      out_path=None,
                      **kwargs):
    """
    :param train_path:      Str path to train dataset
    :param test_path:       Str path to test dataset
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load, one of ['AE', 'OCSVM', 'IF']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """

    print('Loading data...')
    device = dist_util.dev(conf.get('device'))

    # load anomalies to be explained
    if model == 'DDPM' or background == 'diffusion':
        real_data_path = os.path.normpath(conf.real_data_path)
        dataset, X_expl, _ = make_dataset_shap(
            data_path=real_data_path,
            data_name=conf.data_name,
            cat_encoding=conf.cat_encoding,
            num_encoding=conf.num_encoding,
        )
        cat_sizes = dataset.get_category_sizes('test')
        num_numerical_features = dataset.X_num['test'].shape[1] if dataset.X_num is not None else 0

        K = np.array(cat_sizes)
        if len(K) == 0:
            K = np.array([0])
        d_in = np.sum(K) + num_numerical_features
        conf.model_params['d_in'] = d_in

        model_ = get_model(
            conf.model_type,
            conf.model_params,
            num_numerical_features,
            category_sizes=cat_sizes
        )

        model_path = os.path.join(conf.parent_dir, 'model.pt')
        model_.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        model_.to(device)
        model_.eval()

        diffusion = GaussianMultinomialDiffusion(
            K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model_,
            gaussian_loss_type=conf.diffusion_params['gaussian_loss_type'],
            num_timesteps=conf.diffusion_params['num_timesteps'],
            scheduler=conf.diffusion_params['scheduler'],
            device=device,
            inpainting_conf=conf,
            cat_encoding=conf.cat_encoding,
        )
        diffusion.to(device)
        diffusion.eval()
    else:
        cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized')
        X_expl = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / f'anomalies_rand.csv',
                             index_col=None, usecols=cols + ['attackType'], header=0, dtype={'attackType': str, **dtypes})

    # for calculation of reference points
    if background in ['mean', 'kmeans', 'NN']:
        X_train = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / 'train.csv.gz',
                              index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')
        X_train = X_train.sample(frac=0.001, random_state=42)  # sample normal data for kmeans and NN background
    else:
        X_train = pd.DataFrame(np.empty(X_expl.shape), columns=X_expl.columns, index=X_expl.index)

    print('Loading detector...')
    if model == 'AE':
        from anomaly_detection.autoencoder_torch import Autoencoder
        params = {'cpus': 8, 'n_layers': 3, 'n_bottleneck': 32, 'epochs': 10, 'batch_size': 2048, 'verbose': 2,
                  'learning_rate': 0.01, 'n_inputs': 146}  # best params for cidds-ae-16
        detector = Autoencoder(**params)
        detector = detector.load('./xai/outputs/models/cidds/cidds-ae-16_best.pt')
        detector.to(device)
        detector.eval()
    elif model == 'IF':
        import joblib
        detector = joblib.load('./xai/outputs/models/cidds/cidds-if-41_best.pkl')
        detector = DiffusionADWrapper(detector)
    elif model == 'OCSVM':
        import joblib
        detector = joblib.load('./xai/outputs/models/cidds/cidds-oc-12_best.pkl')
        detector = DiffusionADWrapper(detector)
    elif model == 'DDPM':
        detector = diffusion
    else:
        raise ValueError(f"Model {model} not supported!")

    # Generating explanations
    if not os.path.exists(os.path.join(expl_folder, f'{model}_{xai_type}_{background}_{job_name}.csv')):
        print("Generating explanations...")
        out_template = os.path.join(expl_folder, f'{model}_{{}}_{background}_{job_name}.csv')

        if background == 'diffusion':
            predict_fn = functools.partial(
                detector.score_samples,
                is_ordinal=conf.cat_encoding == 'ordinal' or conf.model_type == 'mlp',
                data_name='cidds',
                device=device
            )
        else:
            predict_fn = detector.score_samples

        if xai_type == 'shap':
            import xai.xai_shap

            # get reference points
            if background in ['zeros', 'mean', 'NN', 'single_optimum', 'kmeans', 'kmedoids']:
                if model == 'AE':
                    ref_predict_fn = functools.partial(detector.score_samples, output_to_numpy=False)
                else:
                    ref_predict_fn = predict_fn

                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            columns=X_expl.columns,
                                                            predict_fn=ref_predict_fn)
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_template.format(xai_type),
                                           is_ordinal=conf.cat_encoding == 'ordinal' or conf.model_type == 'mlp',
                                           diffusion_model=diffusion if background == 'diffusion' else None)

        elif xai_type == 'uniform_noise':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_template.format(xai_type))

        else:
            raise ValueError(f'Unknown xai_type: {xai_type}')

    if compare_with_gold_standard:
        print('Evaluating explanations...')
        to_categorical = background != 'diffusion'
        out_dict = evaluate_expls(background=background,
                                  expl_path=f'./xai/outputs/explanation/cidds/{model}_{xai_type}_{background}_{job_name}.csv',
                                  gold_standard_path=f'data/cidds/data_raw/anomalies_rand_expl.csv',
                                  xai_type=xai_type,
                                  model=model,
                                  out_path=out_path,
                                  cat_encoding=conf.cat_encoding,
                                  num_encoding=conf.num_encoding,
                                  to_categorical=to_categorical)
        return out_dict


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """

    parser = ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument(f'--job_name', type=str, default='')
    args_dict = vars(parser.parse_args())
    job_name = args_dict.pop('job_name') if 'job_name' in args_dict else None

    conf_arg = conf_mgt.Default_Conf()
    conf_arg.update(yamlread(args_dict.get('conf_path')))

    # works: ['zeros', 'mean', 'kmeans', 'NN', 'optimized']
    backgrounds = ['diffusion']
    model = 'AE'  # ['AE', 'IF', 'OCSVM']
    xai_type = 'shap'
    compare_with_gold_standard = True
    add_to_summary = True

    expl_folder = './xai/outputs/explanation/cidds'
    out_path = './xai/outputs/explanation/cidds_summary.csv' if add_to_summary else None

    for background in backgrounds:
        explain_anomalies(compare_with_gold_standard=compare_with_gold_standard,
                          expl_folder=expl_folder,
                          xai_type=xai_type,
                          model=model,
                          background=background,
                          job_name=job_name,
                          out_path=out_path,
                          conf=conf_arg)
