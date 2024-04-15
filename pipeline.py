import tomli
import shutil
import os
import argparse

from train import train
import zero
import lib
import torch


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val_optuna', action='store_true', default=False)
    args = parser.parse_args()

    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda')

    timer = zero.Timer()
    timer.run()

    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            data_name=raw_config['data_name'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            device=device,
            validation=args.val_optuna,
            cat_encoding=raw_config['cat_encoding'],
            num_encoding=raw_config['num_encoding']
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'),
              os.path.join(raw_config['real_data_path'], 'info.json'))

    print(f'Elapsed time: {str(timer)}')


if __name__ == '__main__':
    main()
