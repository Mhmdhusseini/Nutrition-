"""
PyTorch neural network definitions for the Smart Diet Planner.

These architectures must exactly match the ones used during training
(see backend.py) so that saved state_dicts load correctly.
"""

import torch.nn as nn


class CaloriesModel(nn.Module):
    """Predicts daily calorie maintenance from user biometrics.

    Input features (7): age, weight, height, gender_encoded, BMI, BMR, activity_level
    Output (1): normalized calorie value
    """

    def __init__(self):
        super(CaloriesModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


class NutritionModel(nn.Module):
    """Predicts nutrient breakdown from caloric value for a specific meal type.

    Input (1): scaled caloric value
    Output (N): scaled nutrient values (protein, carbs, fat, vitamins, minerals, etc.)
    """

    def __init__(self, input_size: int, output_size: int):
        super(NutritionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.fc(x)
