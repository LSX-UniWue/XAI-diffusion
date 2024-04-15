from copy import deepcopy
import torch
import os
import numpy as np
import zero
from lib import FastTensorDataLoader
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, update_ema
from data.data_handling import make_dataset
import lib
import pandas as pd


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


def train(
        parent_dir,
        data_name,
        real_data_path='data/cidds',
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=torch.device('cuda'),
        seed=0,
        validation=False,
        cat_encoding='ordinal',
        num_encoding='zscore',
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    dataset = make_dataset(
        data_path=real_data_path,
        split='train',
        filter_val_anom=validation,
        data_name=data_name,
        cat_encoding=cat_encoding,
        num_encoding=num_encoding,
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0:
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    del dataset

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        cat_encoding=cat_encoding
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))

    del train_loader

    if validation:
        dataset = make_dataset(
            data_path=real_data_path,
            split='val',
            filter_val_anom=validation,
            data_name=data_name,
            cat_encoding=cat_encoding,
            num_encoding=num_encoding,
        )

        y = torch.from_numpy(dataset.y['val'])
        if dataset.X_cat is not None:
            if dataset.X_num is not None:
                dataset = torch.from_numpy(np.concatenate([dataset.X_num['val'], dataset.X_cat['val']], axis=1)).float()
            else:
                dataset = torch.from_numpy(dataset.X_cat['val']).float()
        else:
            dataset = torch.from_numpy(dataset.X_num['val']).float()
        val_loader = FastTensorDataLoader(dataset, y, batch_size=batch_size, shuffle=False)
        del dataset

        diffusion.to(device)
        diffusion.eval()

        total_loss_multi = 0.0
        total_loss_gauss = 0.0
        total_count = 0

        for x, _ in val_loader:
            out = torch.zeros(x.shape[0])
            out_dict = {'y': out}

            x = x.to(device)
            for k in out_dict:
                out_dict[k] = out_dict[k].long().to(device)

            batch_loss_multi, batch_loss_gauss = diffusion.mixed_loss(x, out_dict)

            total_count += len(x)
            total_loss_multi += batch_loss_multi.item() * len(x)
            total_loss_gauss += batch_loss_gauss.item() * len(x)

        loss_multi = np.around(total_loss_multi / total_count, 4)
        loss_gauss = np.around(total_loss_gauss / total_count, 4)

        print("Multinomial Loss during validation was: " + str(loss_multi))
        print("Gaussian Loss during validation was: " + str(loss_gauss))

        report = {
            'multi': loss_multi,
            'gauss': loss_gauss
        }
        lib.dump_json(report, os.path.join(parent_dir, "val_loss.json"))
