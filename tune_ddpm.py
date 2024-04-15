import torch
import lib
import os
import optuna
import shutil
import argparse
from pathlib import Path

from train import train
from pipeline import save_file

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('prefix', type=str)

args = parser.parse_args()
ds_name = args.ds_name
prefix = str(args.prefix)

pipeline = f'pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/')

os.makedirs(exps_path, exist_ok=True)
os.makedirs(parent_path / prefix, exist_ok=True)

param_score_path = parent_path / prefix / 'val_loss_params_score.json'
params_and_score = {}
lib.dump_json(params_and_score, param_score_path)


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def objective(trial):
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 250, 1000])

    params_and_score = lib.load_json(param_score_path)
    params_and_score[trial.number] = {
        'params': {
            'lr': lr,
            'steps': steps,
            'num_timesteps_train': num_timesteps,
        },
        'val_loss': -1
    }
    lib.dump_json(params_and_score, param_score_path)

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['diffusion_params']['num_timesteps'] = num_timesteps

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    raw_config = lib.load_config(f'{exps_path / "config.toml"}')
    device = torch.device(raw_config['device']) if 'device' in raw_config else torch.device('cuda')
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), f'{exps_path / "config.toml"}')

    train(**raw_config['train']['main'],
          **raw_config['diffusion_params'],
          parent_dir=raw_config['parent_dir'],
          data_name=raw_config['data_name'],
          real_data_path=raw_config['real_data_path'],
          model_type=raw_config['model_type'],
          model_params=raw_config['model_params'],
          device=device,
          validation=True,
          cat_encoding=raw_config['cat_encoding'],
          num_encoding=raw_config['num_encoding'])

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'),
              os.path.join(raw_config['real_data_path'], 'info.json'))

    report_path = str(Path(base_config['parent_dir']) / 'val_loss.json')
    report = lib.load_json(report_path)

    shutil.rmtree(exps_path / f"{trial.number}")

    loss = report['multi'] + report['gauss']

    params_and_score = lib.load_json(param_score_path)
    params_and_score[str(trial.number)]['val_loss'] = loss
    lib.dump_json(params_and_score, param_score_path)

    return loss


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)

raw_config = lib.load_config(f'{best_config_path}')
device = torch.device(raw_config['device']) if 'device' in raw_config else torch.device('cuda')
save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), f'{best_config_path}')

train(**raw_config['train']['main'],
      **raw_config['diffusion_params'],
      parent_dir=raw_config['parent_dir'],
      data_name=raw_config['data_name'],
      real_data_path=raw_config['real_data_path'],
      model_type=raw_config['model_type'],
      model_params=raw_config['model_params'],
      device=device,
      validation=False,
      cat_encoding=raw_config['cat_encoding'],
      num_encoding=raw_config['num_encoding'])

save_file(os.path.join(raw_config['parent_dir'], 'info.json'),
          os.path.join(raw_config['real_data_path'], 'info.json'))
