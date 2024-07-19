import torch
import torch.nn as nn
import models.adv_mmvae as adv_mmvae
from torch.utils.tensorboard import SummaryWriter
from utils.constants import REGISTRY_KEYS as RK

import eval.load_model as load_model

def train_md(checkpoint_path:str, config, device, num_epochs=10, lr=0.001):
    model = load_model.load_model_from_checkpoint(checkpoint_path, config)
    data_loader = load_model.create_dataloader(128)
    model.eval()
    writer = SummaryWriter("/mnt/home/taylcard/dev/md_metrics")

    disc = nn.Sequential(
        nn.Linear(256, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64,1),
        nn.Sigmoid()
    )

    disc.to(device)
    model.to(device)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr)
    for epoch in range(num_epochs):
        running_sum = 0
        batch_iteration = 0
        for batch_dict in data_loader:
            batch_iteration += 1
            batch_dict[RK.X] = batch_dict[RK.X].to(device)
            data = batch_dict.get(RK.X)
            data = data.to(device)
            disc.zero_grad()

            forward_outputs = model(batch_dict)

            human_label = torch.zeros(model.hparams.batch_size, 1, device=device)
            mouse_label = torch.ones(model.hparams.batch_size, 1, device=device)

            truth = human_label if batch_dict[RK.EXPERT] == "human" else mouse_label

            disc_prediction = disc(forward_outputs.encoder_act[1])

            loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="mean")
            running_sum += loss_d

            loss_d.backward()
            disc_optimizer.step()

        data_loader.reset()
        print(batch_iteration)
        writer.add_scalar('MD', running_sum / batch_iteration, epoch)