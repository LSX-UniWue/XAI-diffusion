from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion


def get_model(
        model_name,
        model_params,
        n_num_features,
        category_sizes
):
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params, category_sizes=category_sizes)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)
