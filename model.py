import torch
import torch.nn as nn
import torch.nn.functional as F


class NBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NBEATSBlock, self).__init__()
        self.fc_stack = nn.ModuleList(
            [nn.Linear(input_size, hidden_size)]
            + [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for fc in self.fc_stack:
            x = F.relu(fc(x))
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        return backcast, forecast


class NBEATS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_blocks):
        super(NBEATS, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(input_size, hidden_size, output_size, num_layers)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        forecast = torch.zeros(
            x.size(0), self.output_size, device=x.device, dtype=x.dtype
        )

        for block in self.blocks:
            backcast_part, forecast_part = block(x)
            x = x - backcast_part
            forecast += forecast_part
        return forecast
