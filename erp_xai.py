
import os
import functools
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from data.data_handling import make_dataset_shap
from repaint import conf_mgt
from repaint.guided_diffusion import dist_util
from repaint.utils import yamlread
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model
from xai.util import xai_to_categorical, tabular_reference_points
from xai.outputs.models.erp_fraud.util import load_best_detector
from data.erp_fraud.erpDataset import ERPDataset


def get_expl_scores(explanation, gold_standard, dataset, score_type='auc_roc', to_categorical=False):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    scores = []
    for i, row in explanation.iterrows():
        # Explanation values for each feature treated as likelihood of anomalous feature
        #  -aggregated to feature-scores over all feature assignments
        #  -flattened to match shape of y_true
        #  -inverted, so higher score means more anomalous
        if to_categorical:
            y_score = xai_to_categorical(expl_df=pd.DataFrame(explanation.loc[i]).T,
                                         dataset=dataset).values.flatten() * -1
        else:
            y_score = row.values.flatten() * -1
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.loc[i], y_score=y_score))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.loc[i].values.reshape(1, -1), y_score.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   train_path,
                   test_path,
                   gold_standard_path,
                   expl_folder,
                   xai_type,
                   job_name,
                   out_path,
                   data,
                   to_categorical):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(Path(expl_folder) / f'{xai_type}_{background}_{Path(test_path).stem}_{job_name}.csv',
                       header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    # Load gold standard explanations and convert to pd.Series containing
    # expl = expl * -1
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]  # .apply(lambda x: list(x[x.values].index.values), axis=1)
    to_check = data.get_frauds().index.tolist()

    assert len(to_check) == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {len(to_check)}"

    # watch out for naming inconsistency! The dataset=data that get_expl_scores gets is an ERPDataset instance!
    roc_mean, roc_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='auc_roc',
                                        dataset=data,
                                        to_categorical=to_categorical)
    cos_mean, cos_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='cosine_sim',
                                        dataset=data,
                                        to_categorical=to_categorical)

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


def explain_anomalies(train_path,
                      test_path,
                      compare_with_gold_standard,
                      expl_folder,
                      job_name,
                      conf,
                      xai_type='shap',
                      model='AE',
                      numeric_preprocessing='bucket',
                      background='zeros',
                      out_path=None,
                      **kwargs):
    """
    :param train_path:      Str path to train dataset
    :param test_path:       Str path to test dataset
    :param compare_with_gold_standard:  Whether or not to evaluate the explanations vs the gold standard
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load
    :param numeric_preprocessing:   Str type of numeric preprocessing, one of ['buckets', 'log10', 'zscore', 'None']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """
    if model == 'DDPM' or background == 'diffusion':
        real_data_path = os.path.normpath(conf.real_data_path)
        dataset, X_expl, data = make_dataset_shap(
            data_path=real_data_path,
            data_name=conf.data_name,
            cat_encoding=conf.cat_encoding,
            num_encoding=conf.num_encoding,
            train_path=train_path,
            test_path=test_path,
        )
        cat_encoder_ddpm = dataset.cat_transform
        cat_sizes = dataset.get_category_sizes('test')
        num_numerical_features = dataset.X_num['test'].shape[1] if dataset.X_num is not None else 0

        onehot_data = ERPDataset(
            train_path=train_path,
            test_path=test_path,
            numeric_preprocessing=numeric_preprocessing,
            categorical_preprocessing='onehot',
            keep_index=True,
            **kwargs
        )
        X_train, _, _, _, _, _, _, cat_encoder = onehot_data.preprocessed_data.values()
    else:
        data = ERPDataset(
            train_path=train_path,
            test_path=test_path,
            numeric_preprocessing=numeric_preprocessing,
            categorical_preprocessing='ordinal' if 'ordinal' in xai_type else 'onehot',
            keep_index=True,
            **kwargs
        )

        X_train, _, _, _, X_test, y_test, _, _ = data.preprocessed_data.values()
        X_expl = data.get_frauds().sort_index()

    # find gold standard explanations for anomalous cases
    ds_file = Path(test_path).stem + "_expls.csv"
    gold_expl_path = f'data/erp_fraud/{ds_file}'

    print('Loading detector...')
    if model == 'DDPM' or background == 'diffusion':
        device = dist_util.dev(conf.get('device'))

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
        if model == 'DDPM':
            detector = diffusion
        else:
            detector = load_best_detector(model=model, train_path=train_path, test_path=test_path, background=background)
    else:
        detector = load_best_detector(model=model, train_path=train_path, test_path=test_path, background=background)

    if model == 'IF':
        detector.fit(X_train)

    # Generating explanations
    if not os.path.exists(os.path.join(expl_folder, f'{xai_type}_{background}_{Path(test_path).stem}_{job_name}.csv')):
        print("Generating explanations...")
        out_template = os.path.join(expl_folder, f'{{}}_{background}_{Path(test_path).stem}_{job_name}.csv')

        if xai_type == 'shap':
            import xai.xai_shap

            # set the prediction function
            if background == 'diffusion':
                predict_fn = functools.partial(
                    detector.score_samples,
                    is_ordinal=conf.cat_encoding == 'ordinal',
                    cat_encoder=cat_encoder,
                    cat_encoder_ddpm=cat_encoder_ddpm,
                    data_name='erp',
                    device=device
                )
            else:  # simply using detector forward
                predict_fn = detector.score_samples

            if background in ['zeros', 'mean', 'NN', 'single_optimum', 'kmeans']:
                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            columns=X_expl.columns,
                                                            predict_fn=functools.partial(detector.score_samples,
                                                                                         output_to_numpy=False,
                                                                                         invert_score=False))
            elif background == 'diffusion':
                reference_points = pd.DataFrame(np.empty(X_expl.shape), columns=X_expl.columns, index=X_expl.index)
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_template.format(xai_type),
                                           is_ordinal=conf.cat_encoding == 'ordinal',
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
                                  train_path=train_path,
                                  test_path=test_path,
                                  gold_standard_path=gold_expl_path,
                                  expl_folder=expl_folder,
                                  xai_type=xai_type,
                                  job_name=job_name,
                                  out_path=out_path,
                                  data=data,  # ERPDataset class instance
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

    xai_type = 'shap'

    # ['zeros', 'mean', 'NN', 'kmeans', 'single_optimum', 'optimized', 'diffusion']
    backgrounds = ['diffusion']

    model = 'AE'  # ['AE', 'IF', 'OCSVM', 'DDPM']
    train_path = 'data/erp_fraud/normal_2.csv'
    test_path = 'data/erp_fraud/fraud_3.csv'

    compare_with_gold_standard = True
    add_to_summary = True

    expl_folder = './xai/outputs/explanation/erp_fraud'
    out_path = './xai/outputs/explanation/erp_fraud_summary.csv' if add_to_summary else None

    for background in backgrounds:
        explain_anomalies(train_path=train_path,
                          test_path=test_path,
                          compare_with_gold_standard=compare_with_gold_standard,
                          expl_folder=expl_folder,
                          xai_type=xai_type,
                          model=model,
                          numeric_preprocessing='buckets',
                          background=background,
                          job_name=job_name,
                          out_path=out_path,
                          conf=conf_arg)
