from torch import nn


class MLP(nn.Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MLP, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.linear_layers(inputs)