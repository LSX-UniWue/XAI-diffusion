
import tqdm
import numpy as np
import pandas as pd

from data.preprocessing import (CategoricalOneHotPreprocessor,
                                CategoricalOrdinalPreprocessor,
                                NumericalQuantizationPreprocessor)


def scores_to_categorical(data, categories):
    """np.concatenate(data_cat, data[:, 29:])
    Slims a data array by adding column values of rows together for all column pairs in list categories.
    Used for summing up scores that were calculated for one-hot representation of categorical features.
    Gives a score for each categorical feature.
    :param data:        np.array of shape (samples, features) with scores from one-hot features
    :param categories:  list with number of features that were used for one-hot encoding each categorical feature
                        (as given by sklearn.OneHotEncoder.categories_)
    :return:
    """
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    if data.shape[1] > len(categories):  # get all data columns not in categories and append data_cat
        categories_flat = [item for sublist in categories for item in sublist]
        data_cat = np.concatenate((data[:, list(set(range(data.shape[1])) ^ set(categories_flat))], data_cat), axis=1)
    return data_cat


def create_mapping(dataset):
    counter = 0
    mapping_list = []

    num_prep = dataset.preprocessed_data['num_prep']
    cat_prep = dataset.preprocessed_data['cat_prep']

    if isinstance(cat_prep, CategoricalOneHotPreprocessor):
        for cat_mapping in cat_prep.encoder.category_mapping:
            mapping_list.append(list(range(counter, counter + cat_mapping['mapping'].size - 1)))
            counter += cat_mapping['mapping'].size - 1  # -1 because of double nan handling
    elif isinstance(cat_prep, CategoricalOrdinalPreprocessor):
        for _ in dataset.cat_cols:
            mapping_list.append([counter])
            counter += 1
    else:
        raise ValueError(f"Unknown categorical preprocessing: {type(cat_prep).__name__}")

    if isinstance(num_prep, NumericalQuantizationPreprocessor):
        for _ in range(num_prep.encoder.n_bins_.size):
            n_buckets = num_prep.encoder.n_bins + 1
            mapping_list.append(list(range(counter, counter + n_buckets)))
            counter += n_buckets
    else:
        for _ in dataset.num_cols:
            mapping_list.append([counter])
            counter += 1

    return mapping_list


def xai_to_categorical(expl_df, dataset=None):
    """
    Converts XAI scores to categorical values and adds column names

    Example:
    xai_to_categorical(xai_score_path='./scoring/outputs/ERPSim_BSEG_RSEG/pos_shap.csv',
                       out_path='./scoring/outputs/ERPSim_BSEG_RSEG/joint_shap.csv',
                       data_path='./datasets/real/ERPSim/BSEG_RSEG/ERP_Fraud_PJS1920_BSEG_RSEG.csv')
    """
    cat_cols = create_mapping(dataset)

    col_names = dataset.get_column_names()

    expls_joint = scores_to_categorical(expl_df.values, cat_cols)

    return pd.DataFrame(expls_joint, index=expl_df.index, columns=col_names)


def tabular_reference_points(background, X_expl, X_train=None, columns=None, predict_fn=None):
    if background in ['mean', 'NN']:
        assert X_train is not None, f"background '{background}' requires train data as input at variable 'X_train'"
    if background in ['optimized']:
        assert predict_fn is not None, f"background '{background}' requires predict_fn as input"

    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)
        return reference_points

    elif background == 'mean':  # mean training data point for each data point
        reference_points = np.mean(X_train, axis=0).reshape((1, -1)).repeat(X_expl.shape[0], axis=0)
        return reference_points

    elif background == 'NN':  # nearest neighbor in the normal training data for each data point
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl, n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]
        return reference_points

    elif background == 'single_optimum':  # one normal point in the proximity for each data point
        from xai.automated_background_torch import optimize_input_quasi_newton
        reference_points = np.zeros(X_expl.shape)
        for i in tqdm.tqdm(range(X_expl.shape[0]), desc='generating reference points'):
            reference_points[i] = optimize_input_quasi_newton(data_point=X_expl[i].reshape((1, -1)),
                                                              kept_feature_idx=None,
                                                              predict_fn=predict_fn)
        return reference_points

    elif background == 'kmeans':  # kmeans cluster centers of normal data as global background
        from sklearn.cluster import k_means
        centers, _, _ = k_means(X=X_train, n_clusters=5)
        return centers

    else:
        raise ValueError(f"Unknown background: {background}")
