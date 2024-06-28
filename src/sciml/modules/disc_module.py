import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        disc_layers 
    ):
        super().__init__()
        self.disc_layers = disc_layers
        self.disc = self.build_network()

    def build_network(self):
        layers = []
        n_in = self.disc_layers[0]
        for n_out in self.disc_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.Sigmoid(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)