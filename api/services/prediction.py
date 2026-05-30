"""
Inference / prediction logic — ported from the original Streamlit app.

All functions are pure (no Streamlit dependency, no globals).
They receive the ModelStore explicitly so they are easy to test.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from api.services.model_loader import ModelStore


# ── Helper calculations ─────────────────────────────────────────────

def calculate_bmi(weight: float, height: float) -> float:
    """BMI = weight (kg) / height (m)²"""
    return weight / (height ** 2)


def calculate_bmr(age: int, weight: float, height: float, gender: str) -> float:
    """Mifflin-St Jeor / Harris-Benedict BMR estimation."""
    height_cm = height * 100
    if gender == "M":
        return 88.362 + (13.397 * weight) + (4.799 * height_cm) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height_cm) - (4.330 * age)


def get_bmi_category(bmi: float) -> str:
    """Return a human-readable BMI category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# ── Activity level mapping ──────────────────────────────────────────

ACTIVITY_MAP: Dict[str, float] = {
    "Sedentary": 1.2,
    "Active": 1.55,
    "Very Active": 1.725,
}


# ── Calorie prediction & adjustment ────────────────────────────────

def predict_calories(inputs: list, store: ModelStore) -> float:
    """Run the CaloriesModel and denormalize the output."""
    inputs_scaled = store.calories_scaler_X.transform([inputs])
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = store.calories_model(inputs_tensor).item()
    return prediction * store.y_std + store.y_mean


def adjust_calories(tdee: float, goal: str, gender: str) -> float:
    """Apply deficit / surplus based on the user's weight goal."""
    if goal == "Maintain Weight":
        return tdee
    elif goal == "Lose Weight":
        minimum = 1200.0 if gender == "F" else 1500.0
        return max(tdee - 500, minimum)
    elif goal == "Gain Weight":
        return tdee + 500
    return tdee  # fallback


def divide_calories(total: float) -> Tuple[float, float, float]:
    """Split total calories into breakfast (25%), lunch (31%), dinner (35%)."""
    return total * 0.25, total * 0.31, total * 0.35


# ── Nutrient prediction ────────────────────────────────────────────

def predict_nutrients(
    caloric_value: float,
    model: Any,
    scaler_X: Any,
    scaler_y: Any,
) -> np.ndarray:
    """Predict nutrient breakdown for a given caloric intake."""
    scaled = scaler_X.transform([[caloric_value]])
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(tensor).numpy()
    return scaler_y.inverse_transform(prediction)[0]


# ── Recipe selection ────────────────────────────────────────────────

def select_meal_recipes(
    meal_df: pd.DataFrame,
    target_calories: float,
    target_nutrients: Dict[str, float],
    max_attempts: int = 10_000,
    calorie_tolerance: float = 0.15,
) -> Optional[pd.DataFrame]:
    """
    Randomly sample recipe combos and pick the one closest to targets.

    Ported from app.py `select_meal_recipes` — identical algorithm,
    but returns None instead of showing st.warning.
    """

    nutrient_priority = {
        "Protein": 0.15,
        "Carbohydrates": 0.15,
        "Fat": 0.15,
        "default": 0.25,
    }

    def _score(
        actual_cal: float,
        actual_nutrients: Dict[str, float],
        target_cal: float,
        target_nut: Dict[str, float],
    ) -> float:
        scores: List[float] = []
        cal_err = (
            abs(actual_cal - target_cal) / target_cal
            if target_cal != 0
            else (0.0 if actual_cal == 0 else 1.0)
        )
        scores.append(min(cal_err / 0.10, 1.0))
        for nut, value in target_nut.items():
            actual = actual_nutrients.get(nut, 0.0)
            tol = nutrient_priority.get(nut, nutrient_priority["default"])
            err = (
                abs(actual - value) / value
                if value != 0
                else (0.0 if actual == 0 else 1.0)
            )
            scores.append(min(err / tol, 1.0))
        return float(np.mean(scores))

    nutrient_cols = list(target_nutrients.keys())
    required_cols = ["Caloric Value"] + nutrient_cols
    missing = [c for c in required_cols if c not in meal_df.columns]
    if missing:
        return None

    best_score = float("inf")
    best_combo: Optional[pd.DataFrame] = None
    smallest_cal_err = float("inf")
    closest_combo: Optional[pd.DataFrame] = None

    for _ in range(max_attempts):
        n = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
        try:
            sample = meal_df.sample(n=n, replace=False)
        except ValueError:
            continue

        total_cal = sample["Caloric Value"].sum()

        if (target_calories * (1 - calorie_tolerance)) <= total_cal <= (
            target_calories * (1 + calorie_tolerance)
        ):
            total_nuts = sample[nutrient_cols].sum().to_dict()
            score = _score(total_cal, total_nuts, target_calories, target_nutrients)
            if score < best_score:
                best_score = score
                best_combo = sample

        cal_err = abs(total_cal - target_calories)
        if cal_err < smallest_cal_err:
            smallest_cal_err = cal_err
            closest_combo = sample

    return best_combo if best_combo is not None else closest_combo


# ── Orchestrator ────────────────────────────────────────────────────

def generate_plan(
    age: int,
    gender: str,
    weight: float,
    height: float,
    activity_level: str,
    goal: str,
    store: ModelStore,
) -> Dict:
    """
    End-to-end diet plan generation.

    Returns a dict matching the PlanResponse schema.
    """

    # 1. Biometrics
    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(age, weight, height, gender)
    bmi_category = get_bmi_category(bmi)

    # 2. Encode gender for the model
    gender_encoded = store.gender_label_encoder.transform([gender])[0]
    activity_numeric = ACTIVITY_MAP[activity_level]

    # 3. Predict TDEE
    inputs = [age, weight, height, gender_encoded, bmi, bmr, activity_numeric]
    tdee = predict_calories(inputs, store)

    # 4. Adjust for goal
    recommended = adjust_calories(tdee, goal, gender)

    # 5. Split into meals
    bfast_cal, lunch_cal, dinner_cal = divide_calories(recommended)

    # 6. Per-meal nutrients & recipes
    meals_out: Dict[str, Any] = {}
    meal_cals = {"breakfast": bfast_cal, "lunch": lunch_cal, "dinner": dinner_cal}

    for meal, cal in meal_cals.items():
        # Predict nutrients
        nutrients_arr = predict_nutrients(
            cal,
            store.meal_models[meal],
            store.meal_scalers_X[meal],
            store.meal_scalers_y[meal],
        )
        targets = store.meal_targets[meal]
        nutrients_dict = {targets[i]: float(nutrients_arr[i]) for i in range(len(targets))}

        # Select recipes
        recipes_df = select_meal_recipes(
            store.meal_data[meal], cal, nutrients_dict
        )

        # Build recipe list
        recipe_list: List[Dict[str, Any]] = []
        total_cal = 0.0
        if recipes_df is not None and len(recipes_df) > 0:
            for _, row in recipes_df.iterrows():
                recipe_list.append(
                    {
                        "food": str(row.get("food", "Unknown")).title(),
                        "calories": round(float(row.get("Caloric Value", 0)), 1),
                        "protein": round(float(row.get("Protein", 0)), 1),
                        "carbohydrates": round(float(row.get("Carbohydrates", 0)), 1),
                        "fat": round(float(row.get("Fat", 0)), 1),
                    }
                )
            total_cal = float(recipes_df["Caloric Value"].sum())

        # Round predicted nutrients for cleaner output
        rounded_nutrients = {k: round(v, 2) for k, v in nutrients_dict.items()}

        meals_out[meal] = {
            "target_calories": round(cal, 2),
            "recipes": recipe_list,
            "total_calories": round(total_cal, 2),
            "predicted_nutrients": rounded_nutrients,
        }

    return {
        "bmi": round(bmi, 2),
        "bmi_category": bmi_category,
        "bmr": round(bmr, 2),
        "tdee": round(tdee, 2),
        "recommended_calories": round(recommended, 2),
        "goal": goal,
        "meals": meals_out,
    }
