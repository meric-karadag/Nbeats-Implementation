import torch
import torch.nn as nn
import torch.nn.functional as F


class NBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super(NBEATSBlock, self).__init__()
        self.fc_stack = [nn.Linear(input_size, hidden_size), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            self.fc_stack.append(nn.Linear(hidden_size, hidden_size))
            self.fc_stack.append(nn.Dropout(dropout))
        self.fc_stack = nn.ModuleList(self.fc_stack)
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for fc in self.fc_stack:
            x = F.relu(fc(x))
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        return backcast, forecast


class NBEATS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_blocks, dropout=0.0):
        super(NBEATS, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(input_size, hidden_size, output_size, num_layers, dropout=dropout)
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
