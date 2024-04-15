import tqdm
from functools import partial

from scipy.optimize import minimize
import numpy as np
import torch


def optimize_input_quasi_newton(data_point, kept_feature_idx, predict_fn, threshold=None, points_to_keep_distance=(),
                                proximity_weight=0.01, diversity_weight=0.01, device='cpu',
                                batch_norm_regularization=False):
    """
    idea from: http://www.bnikolic.co.uk/blog/pytorch/python/2021/02/28/pytorch-scipyminim.html

    Uses quasi-Newton optimization (Sequential Least Squares Programming) to find optimal input alteration for model
    according to:
    loss = predict_fn(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
           (optionally + delta * negative distance to points that should be avoided [Haldar2021])
    :param data_point:          numpy model input to optimize
    :param kept_feature_idx:    index of feature in data_point to keep, or None for not constraining any feature
                                Can also contain a list of indices to keep
    :param predict_fn:          function of pytorch model to optimize loss for
    :param proximity_weight:    float weight loss factor for proximity to the optimized input
    :param diversity_weight:    float weight loss factor for points_to_keep_distance
    :param points_to_keep_distance: list of numpy data points to keep distance from (distance added to loss function)
    :param batch_norm_regularization: bool whether to add regularization loss for batch norm layers
    :return:                    numpy optimized data_point
    """
    data_point = torch.autograd.Variable(torch.from_numpy(data_point.astype('float32')), requires_grad=True).to(device)
    proximity_weight = torch.Tensor([proximity_weight]).to(device)
    diversity_weight = torch.Tensor([diversity_weight]).to(device)
    if threshold is not None:
        threshold = torch.Tensor([threshold]).to(device)
    zero_elem = torch.tensor(0.0)
    if len(points_to_keep_distance) > 0:
        points_to_keep_distance = torch.tensor(np.concatenate([p.reshape([1, -1]) for p in points_to_keep_distance]),
                                               dtype=torch.float32, device=device)
    else:
        points_to_keep_distance = None

    def val_and_grad(x):
        if batch_norm_regularization:
            pred_loss, intermediate_logits = predict_fn(x, return_pre_norm_logits=True)
        else:
            pred_loss = predict_fn(x)
        if threshold is not None:  # hinge loss for anomaly score
            pred_loss = torch.max(zero_elem, pred_loss - threshold)
        prox_loss = proximity_weight * torch.linalg.vector_norm(data_point - x)
        if points_to_keep_distance is not None:
            divs_loss = diversity_weight * torch.max(
                -1 * torch.norm(points_to_keep_distance - x.repeat(len(points_to_keep_distance), 1), dim=1))
        else:
            divs_loss = 0
        loss = pred_loss + prox_loss + divs_loss
        if batch_norm_regularization:
            # add loss when weights are too far from gaussian of batch norm layers
            batch_norm_loss = torch.tensor(0.0)
            for m in intermediate_logits:
                logits = m['logits']
                # mean = m['mean']
                # var = m['var']
                # abs_normed = torch.abs(logits - mean) / torch.sqrt(var)  # logits are already normed!
                # add loss if logits are outside of 99% range from gaussian with mean and var
                # https://www.mittag-statistik.de/app/quantiles/standardnormal.html
                batch_norm_loss += torch.sum(torch.max(zero_elem, logits - 1.6449))  # 0.005: 2.5758, 0.05: 1.6449
            loss += batch_norm_loss

        loss.backward()
        grad = x.grad
        return loss, grad

    def func(x):
        """scipy needs flattened numpy array with float64, tensorflow tensors with float32"""
        return [vv.cpu().detach().numpy().astype(np.float64).flatten() for vv in
                val_and_grad(torch.tensor(x.reshape([1, -1]), dtype=torch.float32, requires_grad=True))]

    if kept_feature_idx is None:
        constraints = ()
    elif type(kept_feature_idx) == int:
        constraints = {'type': 'eq',
                       'fun': lambda x: x[kept_feature_idx] - data_point.detach().numpy()[:, kept_feature_idx]}
    elif len(kept_feature_idx) != 0:
        kept_feature_idx = np.where(kept_feature_idx)[0]
        constraints = []
        for kept_idx in kept_feature_idx:
            constraints.append(
                {'type': 'eq',
                 'fun': partial(lambda x, idx: x[idx] - data_point.detach().numpy()[:, idx], idx=kept_idx)})
    else:
        constraints = ()

    res = minimize(fun=func,
                   x0=data_point.detach().cpu(),
                   jac=True,
                   method='SLSQP',
                   constraints=constraints)
    opt_input = res.x.astype(np.float32).reshape([1, -1])

    return opt_input


def dynamic_synth_data(sample, maskMatrix, model, predict_fn, background_type, diffusion_model=None):
    """
    Dynamically generate background "deletion" data for each synthetic data sample by minimizing model output.
    :param background_type: 'optimized': Finds most benign inputs through SLSQP while always constraining 1 feature,
                                    takes mean of all benign inputs when generating background for samples where
                                    more then 1 feature needs to be constrained (instead of solving SLSQP again).
    :param sample:                  np.ndarray sample to explain, shape (1, n_features)
    :param maskMatrix:              np.ndarray matrix with features to remove in SHAP sampling process
                                    1 := keep, 0 := optimize/remove
    :param model:                   ml-model to optimize loss for
    :return:                        np.ndarray with synthetic samples, shape maskMatrix.shape

    Example1:
    dynamic_synth_data(sample=to_explain[0].reshape([1, -1]),
                    maskMatrix=maskMatrix,
                    model=load_model('../outputs/models/AE_cat'),
                    background_type='full')

    Example2:
    # integrate into SHAP in shap.explainers.kernel @ KernelExplainer.explain(), right before calling self.run()
    if self.dynamic_background:
        from xai.automated_background import dynamic_synth_data
        self.synth_data, self.fnull = dynamic_synth_data(sample=instance.x,
                                                        maskMatrix=self.maskMatrix,
                                                        model=self.full_model,
                                                        background_type=self.dynamic_background)
        self.expected_value = self.fnull
    """
    assert sample.shape[0] == 1, \
        f"Dynamic background implementation can't explain more then one sample at once, but input had shape {sample.shape}"
    assert maskMatrix.shape[1] == sample.shape[1], \
        f"Dynamic background implementation requires sampling of all features (omitted in SHAP when baseline[i] == sample[i]):\n" \
        f"shapes were maskMatrix: {maskMatrix.shape} and sample: {sample.shape}\n" \
        f"Use of np.inf vector as SHAP baseline is recommended"

    if background_type in ['optimized']:
        # optimize all permutations with 1 kept variable, then aggregate results
        batch_norm_regularization = background_type == 'optimized_batch_norm'
        x_hat = []  # contains optimized feature (row) for each leave-one-out combo of varying features (column)
        # Sequential Least Squares Programming
        for kept_idx in tqdm.tqdm(range(sample.shape[1])):
            x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                     kept_feature_idx=kept_idx,
                                                     predict_fn=model,
                                                     batch_norm_regularization=batch_norm_regularization))
        x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                 kept_feature_idx=None,
                                                 predict_fn=model,
                                                 batch_norm_regularization=batch_norm_regularization))
        x_hat = np.concatenate(x_hat)

        # print("Debugging: ###############################################")
        # print("Old sample MSE:\t\t", MSE(y_true=sample, y_pred=model(sample)))
        # print("New sample MSE:\t\t", MSE(y_true=x_hat[-1].reshape([1, -1]), y_pred=model(x_hat[-1].reshape([1, -1]))))
        # print("Euclidean distance:\t", MSE(y_true=sample, y_pred=x_hat[-1].reshape([1, -1])))

        # Find x_tilde by adding x_hat entries for each feature to keep
        def sum_sample(row):
            S = x_hat[:-1][row == True]
            return ((S.sum(axis=0) + x_hat[-1]) / (S.shape[0] + 1)).reshape([1, -1])

        x_tilde_Sc = []
        for mask in maskMatrix:
            x_tilde_Sc.append(sum_sample(mask))
        x_tilde_Sc = np.concatenate(x_tilde_Sc)
        x_tilde = sample.repeat(maskMatrix.shape[0], axis=0) * maskMatrix + x_tilde_Sc * (1 - maskMatrix)

        fnull = model(torch.tensor(x_hat[-1]).unsqueeze(0)).detach().numpy()
        return x_tilde, fnull

    elif background_type == 'diffusion':
        sample = diffusion_model.to_tensor(sample)
        normal_mask = np.zeros(sample.shape[1])

        # bring categorical part of mask to ohe length
        normal_mask_num = normal_mask[:diffusion_model.num_numerical_features]
        normal_mask_cat = [num for num, count in
                            zip(normal_mask[diffusion_model.num_numerical_features:], diffusion_model._denoise_fn.category_sizes)
                            for _ in range(count)]
        normal_mask_ohe = [np.concatenate((normal_mask_num, normal_mask_cat))]
        normal_mask = torch.Tensor(normal_mask_ohe)

        shap_masks_ohe = []
        for mask in maskMatrix:
            mask_num = mask[:diffusion_model.num_numerical_features]
            mask_cat = [num for num, count in zip(mask[diffusion_model.num_numerical_features:], diffusion_model._denoise_fn.category_sizes)
                        for _ in range(count)]
            mask_ohe = np.concatenate((mask_num, mask_cat))
            shap_masks_ohe.append(mask_ohe)
        shap_masks = torch.Tensor(shap_masks_ohe)

        # concat normal + other masks, split again after calculating results
        masks = torch.cat((shap_masks, normal_mask), dim=0)

        inpainting_res = diffusion_model.inpaint(x=sample, masks=masks, ordinal=(diffusion_model.cat_encoding == 'ordinal' or diffusion_model.inpainting_conf.model_type == 'mlp'))["sample"]

        all_normal = inpainting_res[-1:, :]
        fnull = predict_fn(all_normal)

        synth_data = inpainting_res[:-1, :].detach().cpu().numpy()

        return synth_data, fnull

    else:
        raise ValueError(
            f"Variable 'background_type' needs to be one of ['optimized', 'diffusion'], but was {background_type}")
