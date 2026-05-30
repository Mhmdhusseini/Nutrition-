"""
Pydantic models for request validation and response serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────

class Gender(str, Enum):
    male = "M"
    female = "F"


class ActivityLevel(str, Enum):
    sedentary = "Sedentary"
    active = "Active"
    very_active = "Very Active"


class WeightGoal(str, Enum):
    maintain = "Maintain Weight"
    lose = "Lose Weight"
    gain = "Gain Weight"


# ── Request ─────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    """User biometric and goal data for diet plan generation."""

    age: int = Field(..., ge=5, le=90, description="Age in years (5–90)")
    gender: Gender = Field(..., description="Biological sex: 'M' or 'F'")
    weight: float = Field(
        ..., gt=12, lt=100, description="Body weight in kg (12–100)"
    )
    height: float = Field(
        ..., gt=0.86, lt=2.0, description="Height in metres (0.86–2.0)"
    )
    activity_level: ActivityLevel = Field(
        ..., description="Activity level: 'Sedentary', 'Active', or 'Very Active'"
    )
    goal: WeightGoal = Field(
        ..., description="Weight goal: 'Maintain Weight', 'Lose Weight', or 'Gain Weight'"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 25,
                    "gender": "M",
                    "weight": 75.0,
                    "height": 1.78,
                    "activity_level": "Active",
                    "goal": "Lose Weight",
                }
            ]
        }
    }


# ── Response ────────────────────────────────────────────────────────

class RecipeItem(BaseModel):
    """A single food item / recipe."""

    food: str
    calories: float
    protein: float
    carbohydrates: float
    fat: float


class MealPlan(BaseModel):
    """Nutrition plan for a single meal (breakfast / lunch / dinner)."""

    target_calories: float
    recipes: List[RecipeItem]
    total_calories: float
    predicted_nutrients: Dict[str, float]


class PlanResponse(BaseModel):
    """Complete diet plan returned to the client."""

    bmi: float
    bmi_category: str
    bmr: float
    tdee: float
    recommended_calories: float
    goal: str
    meals: Dict[str, MealPlan]
