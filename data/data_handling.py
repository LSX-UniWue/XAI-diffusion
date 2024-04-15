import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from data.cidds.util import get_cols_and_dtypes
from data.erp_fraud.erpDatasetTabDDPM import ERPDatasetTabDDPM
import lib
from tab_ddpm import ohe_to_categories


def make_dataset_shap(data_path: str, data_name: str, num_encoding, cat_encoding,
                      train_path='data/erp_fraud/normal_2.csv',
                      test_path='data/erp_fraud/fraud_3.csv'):
    if data_name == 'cidds':
        cols, dtypes = get_cols_and_dtypes(cat_encoding=cat_encoding, num_encoding=num_encoding)
        X_cat, X_num, y = {}, {}, {}

        X = pd.read_csv(
            Path(data_path) / 'data_prep' / 'anomalies_rand.csv', index_col=None, usecols=cols, header=0,
            dtype={**dtypes}
        )[cols]

        # special procedure to load given preprocessed anomalies where a few attributes are already one-hot, some not
        # transform isWeekday, isSYN, isACK, isFIN, isURG, isPSH, isRES into ohe
        columns_to_transform = ['isWeekday', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
        new_columns = []

        for column in columns_to_transform:
            new_columns.append(f'{column} 0')
            new_columns.append(f'{column} 1')

        new_df = pd.DataFrame(0, columns=new_columns, index=X.index)

        for column in columns_to_transform:
            new_df[f'{column} 0'] = (X[column] == 0).astype(int)
            new_df[f'{column} 1'] = (X[column] == 1).astype(int)

        weekday0_df = new_df.pop('isWeekday 0')
        weekday1_df = new_df.pop('isWeekday 1')

        X = pd.concat([weekday0_df, weekday1_df, X, new_df], axis=1)
        X = X.drop(['isWeekday', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES'], axis=1)

        X = ohe_to_categories(torch.tensor(X.values), np.array([2, 5, 4, 3, 36, 19, 4, 36, 19, 4, 4, 5, 2, 2, 2, 2, 2, 2])).numpy()

        X_num = None
        X_cat['test'] = X

        X = pd.DataFrame(X)

        y['test'] = None
        cat_transform = None
        num_transform = None

        if num_encoding == 'zscore':
            means_stds = lib.load_json(os.path.join(data_path, "data_prep/mean_std_dict.json"))
        else:
            means_stds = None
        dataset = None
    elif data_name == 'erp':
        dataset = ERPDatasetTabDDPM(
            numeric_preprocessing='buckets' if num_encoding == 'quantized' else num_encoding,
            categorical_preprocessing=cat_encoding, train_path=train_path, test_path=test_path
        )
        _, X_cat_train, X_num_train, _, _, _, _, _, _, X_cat_test, X_num_test, y_test, num_transform, cat_transform = dataset.preprocessed_data.values()

        X_cat = {
            'train': X_cat_train,
            'test': X_cat_test.to_numpy(),
        }
        if X_num_train is None:
            X_num = None
        else:
            X_num = {
                'train': X_num_train,
                'test': X_num_test.to_numpy() if X_num_test is not None else X_num_test,
            }

        y_test = [0 if i == 'NonFraud' else 1 for i in y_test]
        y = {
            'test': np.asarray(y_test, dtype=np.int8),
        }

        X = dataset.get_frauds().sort_index()

        means_stds = None
    else:
        raise ValueError(
            f"Variable data_name needs to be one of "
            f"['erp', 'cidds'] but was: {data_name}"
        )

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes'),
        cat_transform=cat_transform,
        num_transform=num_transform,
        data_name=data_name,
        means_stds=means_stds,
    )

    return D, X, dataset


def make_dataset(data_path: str, data_name: str, split='train', synth_path=None, filter_val_anom=False,
                 cat_encoding='ordinal', num_encoding='zscore'):
    if split not in ['train', 'val', 'test']:
        raise ValueError(
            f"Variable split needs to be one of "
            f"['train', 'val', 'test'] but was: {split}"
        )
    if data_name == 'erp':
        dataset = ERPDatasetTabDDPM(
            synth_path=synth_path, filter_val_anom=filter_val_anom,
            numeric_preprocessing='buckets' if num_encoding == 'quantized' else num_encoding,
            categorical_preprocessing=cat_encoding
        )
        _, X_cat_train, X_num_train, y_train, _, X_cat_eval, X_num_eval, y_eval, _, X_cat_test, X_num_test, y_test, num_transform, cat_transform = dataset.preprocessed_data.values()

        X_cat_train = X_cat_train.to_numpy()
        X_num_train = X_num_train.to_numpy() if X_num_train is not None else X_num_train
        X_cat = {
            'train': X_cat_train,
            'val': X_cat_eval.to_numpy(),
            'test': X_cat_test.to_numpy(),
        }
        if X_num_train is None:
            X_num = None
        else:
            X_num = {
                'train': X_num_train,
                'val': X_num_eval.to_numpy() if X_num_eval is not None else X_num_eval,
                'test': X_num_test.to_numpy() if X_num_test is not None else X_num_test,
            }

        y_train = np.zeros((len(y_train),), dtype=np.int8)
        y_eval = [0 if i == 'NonFraud' else 1 for i in y_eval]
        y_test = [0 if i == 'NonFraud' else 1 for i in y_test]

        y = {
            'train': y_train,
            'val': np.asarray(y_eval, dtype=np.int8),
            'test': np.asarray(y_test, dtype=np.int8),
        }

        means_stds = None

    elif data_name == 'cidds':
        cols, dtypes = get_cols_and_dtypes(cat_encoding=cat_encoding, num_encoding=num_encoding)
        num_cols = ['Daytime', 'Duration', 'Src Conns', 'Dst Conns', 'Packets', 'Bytes']
        X_cat, X_num, y = {}, {}, {}

        X_split = pd.read_csv(
            Path(data_path) / 'data_prep' / f'{split}.csv.gz', compression='gzip', index_col=None, usecols=cols + ['isNormal'], header=0,
            dtype={'isNormal': np.int8, **dtypes}, encoding='UTF-8')[cols + ['isNormal']]
        print('loaded', split, 'split')

        if split == 'val' and filter_val_anom:
            X_split = X_split[X_split['isNormal'] == 1]

        y_split = 1 - X_split.pop('isNormal')

        if cat_encoding == 'onehot':
            # special procedure to load given preprocessed data where a few attributes are already one-hot, some not
            # transform isWeekday, isSYN, isACK, isFIN, isURG, isPSH, isRES into ohe
            columns_to_transform = ['isWeekday', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
            new_columns = []

            for column in columns_to_transform:
                new_columns.append(f'{column} 0')
                new_columns.append(f'{column} 1')

            new_df = pd.DataFrame(0, columns=new_columns, index=X_split.index)

            for column in columns_to_transform:
                new_df[f'{column} 0'] = (X_split[column] == 0).astype(int)
                new_df[f'{column} 1'] = (X_split[column] == 1).astype(int)

            weekday0_df = new_df.pop('isWeekday 0')
            weekday1_df = new_df.pop('isWeekday 1')

            X_split = pd.concat([weekday0_df, weekday1_df, X_split, new_df], axis=1)
            X_split = X_split.drop(['isWeekday', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES'], axis=1)

        if num_encoding == 'zscore':
            X_num_split = pd.concat([X_split.pop(x) for x in num_cols], axis=1)
            X_num[split] = X_num_split.to_numpy()
        elif num_encoding == 'quantized':
            X_num = None

        X_cat_split = X_split
        X_cat[split] = X_cat_split.to_numpy()

        y[split] = y_split.to_numpy()

        cat_transform = None
        num_transform = None

        if num_encoding == 'zscore':
            means_stds = lib.load_json(os.path.join(data_path, "data_prep/mean_std_dict.json"))
        else:
            means_stds = None

    else:
        raise ValueError(
            f"Variable data_name needs to be one of "
            f"['erp', 'cidds'] but was: {data_name}"
        )

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes'),
        cat_transform=cat_transform,
        num_transform=num_transform,
        data_name=data_name,
        means_stds=means_stds,
    )

    return D
