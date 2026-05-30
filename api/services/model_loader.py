"""
Startup model loader — loads all trained artifacts once and keeps them in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import torch

from api.models.nn_models import CaloriesModel, NutritionModel

logger = logging.getLogger(__name__)

# Resolve the project root (one level above the `api/` package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelStore:
    """Singleton-ish container that holds every trained artifact in memory."""

    # ── Calories model ──────────────────────────────────────────────
    calories_model: CaloriesModel
    calories_scaler_X: Any  # sklearn StandardScaler
    gender_label_encoder: Any  # sklearn LabelEncoder
    y_mean: float
    y_std: float

    # ── Per-meal models ─────────────────────────────────────────────
    meal_models: Dict[str, NutritionModel]
    meal_scalers_X: Dict[str, Any]
    meal_scalers_y: Dict[str, Any]
    meal_targets: Dict[str, List[str]]
    meal_data: Dict[str, pd.DataFrame]

    _loaded: bool = False

    def load(self) -> None:
        """Load all .pth / .pkl files from the project root."""

        if self._loaded:
            logger.info("Models already loaded — skipping.")
            return

        root = _PROJECT_ROOT
        logger.info("Loading models from %s …", root)

        # ── Calories model ──────────────────────────────────────────
        self.calories_model = CaloriesModel()
        self.calories_model.load_state_dict(
            torch.load(root / "models" / "calories_model.pth", map_location="cpu", weights_only=True)
        )
        self.calories_model.eval()

        self.calories_scaler_X = joblib.load(root / "models" / "scalers" / "calories_scaler_X.pkl")
        self.gender_label_encoder = joblib.load(root / "models" / "encoders" / "gender_label_encoder.pkl")

        y_scaler_params = joblib.load(root / "models" / "scalers" / "calories_y_scaler_params.pkl")
        self.y_mean = float(y_scaler_params["mean"])
        self.y_std = float(y_scaler_params["std"])

        # ── Per-meal models ─────────────────────────────────────────
        meals = ["breakfast", "lunch", "dinner"]
        self.meal_models = {}
        self.meal_scalers_X = {}
        self.meal_scalers_y = {}
        self.meal_targets = {}
        self.meal_data = {}

        for meal in meals:
            targets = joblib.load(root / "models" / f"{meal}_targets.pkl")
            model = NutritionModel(input_size=1, output_size=len(targets))
            model.load_state_dict(
                torch.load(root / "models" / f"{meal}_nutrient_model.pth", map_location="cpu", weights_only=True)
            )
            model.eval()

            self.meal_models[meal] = model
            self.meal_scalers_X[meal] = joblib.load(root / "models" / "scalers" / f"{meal}_scaler_X.pkl")
            self.meal_scalers_y[meal] = joblib.load(root / "models" / "scalers" / f"{meal}_scaler_y.pkl")
            self.meal_targets[meal] = targets
            self.meal_data[meal] = pd.read_pickle(root / "models" / f"{meal}_data.pkl")
            logger.info("  ✓ %s model loaded (%d nutrient targets)", meal, len(targets))

        self._loaded = True
        logger.info("All models loaded successfully.")


# Module-level singleton
store = ModelStore()
