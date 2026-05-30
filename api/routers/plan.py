"""
Router for diet plan generation.
"""

from fastapi import APIRouter, HTTPException

from api.schemas.schemas import PlanRequest, PlanResponse
from api.services.model_loader import store
from api.services.prediction import generate_plan

router = APIRouter(tags=["Diet Plan"])


@router.post(
    "/generate-plan",
    response_model=PlanResponse,
    summary="Generate a personalised diet plan",
    description=(
        "Accepts user biometric data and a weight goal, then returns a "
        "complete daily nutrition plan with predicted calories, BMI, BMR, TDEE, "
        "and meal-specific recipe recommendations for breakfast, lunch, and dinner."
    ),
)
async def create_plan(request: PlanRequest):
    """Generate a complete diet plan based on user data."""
    try:
        result = generate_plan(
            age=request.age,
            gender=request.gender.value,
            weight=request.weight,
            height=request.height,
            activity_level=request.activity_level.value,
            goal=request.goal.value,
            store=store,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
