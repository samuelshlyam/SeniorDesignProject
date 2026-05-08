import torch
import torch.nn as nn


class VolatilitySurfaceCNN(nn.Module):
    def __init__(self, param_bounds, context_size=2):
        super().__init__()
        num_parameters = len(param_bounds)

        self.register_buffer(
            'param_mins',
            torch.tensor([b[0] for b in param_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            'param_maxs',
            torch.tensor([b[1] for b in param_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            'param_ranges',
            self.param_maxs - self.param_mins
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # 32 channels * 8 maturity * 11 strikes + context_size (sentiment/articles)
        self.flattened_size = 32 * 8 * 11 + context_size

        self.dense_block = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Linear(128, num_parameters),
            nn.Sigmoid()
        )

    def forward(self, iv_surface, context_data):
        features = self.conv_block(iv_surface)
        features_flat = features.view(features.size(0), -1)
        # Combine IV grid features with the sentiment/articles context values
        combined = torch.cat((features_flat, context_data), dim=1)

        out_0_to_1 = self.dense_block(combined)
        real_world_params = (out_0_to_1 * self.param_ranges) + self.param_mins

        return out_0_to_1, real_world_params
