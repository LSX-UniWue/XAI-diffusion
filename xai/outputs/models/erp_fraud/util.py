
import pickle
from pathlib import Path

from anomaly_detection.autoencoder_torch import Autoencoder
from anomaly_detection.diffusion_wrapper import DiffusionADWrapper


def load_best_detector(model, train_path, test_path, model_folder='./xai/outputs/models/erp_fraud/', background=None):
    if model == 'AE':
        if 'normal_2' in train_path:
            detector = Autoencoder(**pickle.load(open(Path(model_folder) / 'best_params/AE_session2.p', 'rb')))
            detector = detector.load(Path(model_folder) / 'AE_session2_torch', only_model=True)
        else:
            raise ValueError("Unknown train and test dataset combination.")
    elif model == 'IF':
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(**pickle.load(open(Path(model_folder) / 'best_params/IF.p', 'rb')))
        if background == 'diffusion':
            detector = DiffusionADWrapper(detector)
    elif model == 'OCSVM':
        import joblib
        detector = joblib.load(f'./xai/outputs/models/erp_fraud/OC_SVM_session2.pkl')
        if background == 'diffusion':
            detector = DiffusionADWrapper(detector)
    else:
        raise ValueError(f"Expected 'model' to be one of ['OCSVM', 'AE', 'IF'], but was {model}")

    return detector


class DaskOCSVM:
    """Small wrapper to trick dask_ml into parallelizing anomaly detection methods"""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.score_samples(X)
