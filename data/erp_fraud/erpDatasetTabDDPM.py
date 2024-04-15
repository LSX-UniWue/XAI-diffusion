from pathlib import Path
import pandas as pd

from data.preprocessing_tabddpm import (CategoricalOneHotPreprocessor,
                                        CategoricalOrdinalPreprocessor,
                                        NumericalLogPreprocessor,
                                        NumericalZscorePreprocessor,
                                        NumericalQuantizationPreprocessor,
                                        NumericalMinMaxPreprocessor)


class ERPDatasetTabDDPM:
    """
    Dataset class for the ERP datasets used for TabDDPM.
    """

    def __init__(
            self,
            train_path='data/erp_fraud/normal_2.csv',
            test_path='data/erp_fraud/fraud_3.csv',
            eval_path='data/erp_fraud/fraud_2.csv',
            synth_path=None,
            split_id=None,
            numeric_preprocessing='zscore',
            categorical_preprocessing='ordinal',
            language='ger',  # 'ger' or 'eng'
            filter_val_anom=False,
            **kwargs    # nan_buckets and other preprocessor arguments belong into **kwargs
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.eval_path = eval_path
        self.synth_path = synth_path
        self.split_id = split_id

        self.metadata_path = Path(train_path).parent / 'column_information.csv'
        self.numeric_preprocessing = numeric_preprocessing
        self.categorical_preprocessing = categorical_preprocessing

        self.X_train_raw = None
        self.X_eval_raw = None
        self.X_test_raw = None
        self.y_train_raw = None
        self.y_eval_raw = None
        self.y_test_raw = None

        self.cat_encoder = None
        self.num_encoder = None

        self.language = language
        self.filter_val_anom = filter_val_anom

        self.num_cols, self.cat_cols = self.get_column_dtypes(self.language)

        self.preprocessed_data = self.load_and_preprocess(numeric_preprocessing=numeric_preprocessing,
                                                          categorical_preprocessing=categorical_preprocessing,
                                                          **kwargs)

    def load_and_preprocess(self,
                            numeric_preprocessing,
                            categorical_preprocessing,
                            keep_notes=False,
                            keep_index=False,
                            keep_label=False,
                            keep_original_numeric=False,
                            **kwargs):
        """
        Loads data with different numerical preprocessing techniques.

        :param numeric_preprocessing:       One of ['log10', 'zscore', 'buckets', 'minmax', None]
        :param categorical_preprocessing:   One of ['onehot', None]
        :param keep_notes:                  Keeps note columns in data (Default: False)
        :param keep_index:                  Keeps original indices of eval and test data (Default: False)
        :param keep_label:                  Keeps the gold standard label in the X_data (Default: False)
        :param keep_original_numeric:       Duplicates numeric columns before Preprocessing, appends '_orig' to column
                                            names
        :param tabddpm_eval:                Loads synthetic data as train set instead of real data (Default: False)
        :param kwargs:                      Given to Preprocessors (first option is default):
                                                NumericalLogPreprocessor
                                                    nan_bucket: [False, True]
                                                NumericalZscorePreprocessor
                                                    nan_bucket: [False, True]
                                                NumericalMinMaxPreprocessor
                                                    nan_bucket: [False, True]
                                                NumericalQuantizationPreprocessor
                                                    encode: ['onehot', 'ordinal']

        :return:                            Dict containing X_train/eval/test, y_train/test/eval and both preprocessors
        """
        print(f"Loading and preprocessing... ")

        # initializing raw data
        self.X_train_raw, self.X_eval_raw, self.X_test_raw = self._load_erp_splits()

        if self.eval_path or self.split_id:
            self.y_train_raw, self.y_eval_raw, self.y_test_raw = \
                self.X_train_raw["Label"], self.X_eval_raw["Label"], self.X_test_raw["Label"]
        else:
            self.y_train_raw, self.y_test_raw = self.X_train_raw["Label"], self.X_test_raw["Label"]

        # initializing preprocessors:
        if categorical_preprocessing == 'onehot':
            cat_preprocessor = CategoricalOneHotPreprocessor()
        elif categorical_preprocessing == 'onehot_no_zero':
            cat_preprocessor = CategoricalOneHotPreprocessor(off_value=-1)
        elif categorical_preprocessing == 'ordinal':
            cat_preprocessor = CategoricalOrdinalPreprocessor()
        elif categorical_preprocessing == 'None':
            cat_preprocessor = None
        else:
            raise ValueError(
                f"Variable categorical_preprocessing needs to be one of "
                f"['onehot', 'None'] but was: {categorical_preprocessing}")

        if numeric_preprocessing == 'log10':
            num_preprocessor = NumericalLogPreprocessor(**kwargs)
        elif numeric_preprocessing == 'zscore':
            num_preprocessor = NumericalZscorePreprocessor(**kwargs)
        elif numeric_preprocessing == 'minmax':
            num_preprocessor = NumericalMinMaxPreprocessor(**kwargs)
        elif numeric_preprocessing == 'buckets':
            num_preprocessor = NumericalQuantizationPreprocessor(encode='ordinal', **kwargs)
        elif numeric_preprocessing == 'None':
            num_preprocessor = None
        else:
            raise ValueError(
                f"Variable numeric_preprocessing needs to be one of "
                f"['log10', 'zscore', 'buckets', 'minmax', 'None'] but was: {numeric_preprocessing}")

        # Preprocessing of train-eval-test data
        X_train, X_cat_train, X_num_train = self._preprocessing(
            data=self.X_train_raw,
            num_preprocessor=num_preprocessor,
            cat_preprocessor=cat_preprocessor,
            fit_new=True,
            split="train",
            keep_original_numeric=keep_original_numeric
        )

        if self.X_eval_raw is not None:
            X_eval, X_cat_eval, X_num_eval = self._preprocessing(
                data=self.X_eval_raw,
                num_preprocessor=num_preprocessor,
                cat_preprocessor=cat_preprocessor,
                fit_new=False,
                split="eval",
                keep_original_numeric=keep_original_numeric
            )
        else:
            X_eval, X_cat_eval, X_num_eval = None, None, None

        X_test, X_cat_test, X_num_test = self._preprocessing(
            data=self.X_test_raw,
            num_preprocessor=num_preprocessor,
            cat_preprocessor=cat_preprocessor,
            fit_new=False,
            split="test",
            keep_original_numeric=keep_original_numeric
        )

        note_cols = ["Belegnummer", "Position", "Transaktionsart", "Erfassungsuhrzeit"]
        if not keep_notes:
            X_train = X_train.drop(note_cols, axis=1)
            if X_eval is not None:
                X_eval = X_eval.drop(note_cols, axis=1)
            X_test = X_test.drop(note_cols, axis=1)
        else:  # Reorder columns to have notes in front
            new_order = note_cols + [col for col in X_train.columns if col not in note_cols]
            X_train = X_train[new_order]
            if X_eval is not None:
                X_eval = X_eval[new_order]
            X_test = X_test[new_order]

        if not keep_index:
            X_train = X_train.reset_index(drop=True)
            if X_eval is not None:
                X_eval = X_eval.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

        y_train = X_train["Label"]

        if X_eval is not None:
            y_eval = X_eval["Label"]
        else:
            y_eval = None
        y_test = X_test["Label"]

        # Drop Labels
        if not keep_label:
            X_train = X_train.drop(["Label"], axis=1)
            if X_eval is not None:
                X_eval = X_eval.drop(["Label"], axis=1)
            X_test = X_test.drop(["Label"], axis=1)

        return {"X_train": X_train,
                "X_cat_train": X_cat_train,
                "X_num_train": X_num_train,
                "y_train": y_train,
                "X_eval": X_eval,
                "X_cat_eval": X_cat_eval,
                "X_num_eval": X_num_eval,
                "y_eval": y_eval,
                "X_test": X_test,
                "X_cat_test": X_cat_test,
                "X_num_test": X_num_test,
                "y_test": y_test,
                "num_prep": num_preprocessor,
                "cat_prep": cat_preprocessor}

    def _preprocessing(self,
                       data,  # train, test or eval data
                       num_preprocessor,
                       cat_preprocessor,
                       fit_new,
                       split,
                       keep_original_numeric=False):
        """
        One-hot encoding and standardization for BSEG_RSEG.
        """

        data_cat, data_num = None, None

        # 1) Numerical Preprocessing
        data_num = data[self.num_cols]

        if keep_original_numeric:
            data = data.rename({name: name + '_orig' for name in self.num_cols}, axis=1)
        else:
            data = data.drop(self.num_cols, axis=1)

        if num_preprocessor:
            if fit_new:
                num_preprocessor.fit(data_num)
                self.num_encoder = num_preprocessor

            data_idx = data_num.index
            num_trans_dict = num_preprocessor.transform(data_num.reset_index(drop=True))
            data_num = num_trans_dict['num_data']
            nan_attributes = num_trans_dict['nan_attributes']

            data_num.index = data_idx
            if nan_attributes is not None:
                nan_attributes.index = data_idx

        data = data.join(data_num)

        # 2) Categorical Preprocessing
        data[self.cat_cols] = data[self.cat_cols].astype(str)

        if cat_preprocessor:
            data_cat = data[self.cat_cols]

            # append nan_bucket attributes, created during numerical preprocessing
            if num_preprocessor and (nan_attributes is not None):
                data_cat = data_cat.join(nan_attributes)
            data = data.drop(self.cat_cols, axis=1)

            if fit_new:
                cat_preprocessor.fit(data_cat)
                self.cat_encoder = cat_preprocessor

            data_idx = data_cat.index
            data_cat = cat_preprocessor.transform(data_cat.reset_index(drop=True))

            # replace unknown values, set by transform(), by max_id + 1 of each column for index_to_log_one_hot() to work later on
            if split == 'train':
                self.cat_max_values = data_cat.max().tolist()
            else:
                for i, column in enumerate(data_cat):
                    data_cat.loc[data_cat[column] == self.cat_encoder.unknown_value, column] = self.cat_max_values[i] + 1.0

            data_cat.index = data_idx
            data = data.join(data_cat)

        # reordering of dataframe columns
        cols = data.columns.tolist()
        other, num, cat = cols[:5], cols[5:15], cols[15:]
        new_cols = other + cat + num
        data = data[new_cols]

        if self.numeric_preprocessing == 'buckets':
            data_cat = data_cat.join(data_num)
            data_num = None

        return data, data_cat, data_num

    def _load_erp_splits(self):
        """
        Loads the erp system datasets joined together from BSEG and RSEG tables into train, eval and test datasets.
        If there is no path to an eval dataset but a split_od, the test dataset is split at that split_id.
        Else there is no eval dataset.
        """
        column_info = pd.read_csv(self.metadata_path, index_col=0, header=0).T

        benign = pd.read_csv(self.train_path, encoding='ISO-8859-1')

        benign.columns = column_info[self.language].values.T.reshape(column_info[self.language].shape[0])

        X_train = benign

        if self.eval_path:
            fraud1 = pd.read_csv(self.eval_path, encoding='ISO-8859-1')
            fraud1.columns = column_info[self.language].values.T.reshape(column_info[self.language].shape[0])

            fraud2 = pd.read_csv(self.test_path, encoding='ISO-8859-1')
            fraud2.columns = column_info[self.language].values.T.reshape(column_info[self.language].shape[0])

            X_eval = fraud1
            if self.filter_val_anom:
                # drop anomaly samples from eval dataset
                X_eval = X_eval.drop(X_eval[X_eval.Label != 'NonFraud'].index)

            X_test = fraud2
        elif self.split_id:  # elif not self.eval_path and self.split_id
            fraud = pd.read_csv(self.test_path, encoding='ISO-8859-1')
            fraud.columns = column_info[self.language].values.T.reshape(column_info[self.language].shape[0])

            # splitting by split id, default: 50%-50%, making sure to not cut single accounting documents
            X_eval = fraud.iloc[self.split_id:]  # default: last 50 % as eval data (18 frauds)
            X_test = fraud.iloc[:self.split_id]  # default: first 50 % as test data (6 frauds)
        else:
            fraud = pd.read_csv(self.test_path, encoding='ISO-8859-1')
            fraud.columns = column_info[self.language].values.T.reshape(column_info[self.language].shape[0])

            X_eval = None
            X_test = fraud

        return X_train, X_eval, X_test

    def adjust_labels_for_task(self, task=None, labels=None):
        if task:
            labels = self.preprocessed_data[f"y_{task}"].copy()
        labels[~labels.str.startswith("Fraud_")] = "NonFraud"
        return labels

    def get_frauds(self, events=False):
        """
        Returns a pandas DataFrame containing the frauds (or frauds and events) of the test dataset.
        """
        labels = self.preprocessed_data["y_test"].copy()
        if events:
            labels[(~labels.str.startswith("Fraud_")) & (~labels.str.startswith("Event_"))] = "NonFraud"
        else:
            labels[~labels.str.startswith("Fraud_")] = "NonFraud"
        frauds = labels[labels != "NonFraud"].index.values
        return self.preprocessed_data['X_test'].loc[frauds]

    def get_preprocessors(self):
        return self.preprocessed_data["num_prep"], self.preprocessed_data["cat_prep"]

    def get_column_dtypes(self, language):
        """
        :return: Two lists of column headers for categorical and numerical columns.
        """
        column_info = pd.read_csv(self.metadata_path, index_col=0, header=0).T
        cat_cols = column_info[column_info['cat'].astype(float) == 1][language].values
        num_cols = column_info[column_info['num'].astype(float) == 1][language].values
        return num_cols, cat_cols

    def get_column_names(self):
        # Note and label cols are not language specific and named the same in both German and English
        note_and_label_cols = ["Belegnummer", "Position", "Transaktionsart", "Erfassungsuhrzeit", "Label"]
        return self.X_train_raw.drop(note_and_label_cols, axis=1).columns.values

    def get_dataset_information(self):
        return {'train_path': self.train_path,
                'test_path': self.test_path,
                'eval_path': self.eval_path,
                'split_id': self.split_id,
                'metadata_path': self.metadata_path}

    def inverse_transform(self, data):
        """
        Inverse transforms the given data from preprocessed into a pre-preprocessing (raw) format.

        :param data: pandas DataFrame containing the data to be inverse transformed
        """
        pass
