# Smart Diet Planner API

A production-ready FastAPI REST API that generates personalised daily nutrition plans. Powered by PyTorch models trained on nutritional datasets.

## Features
- **Calorie prediction** based on user biometrics (age, gender, weight, height, activity level).
- **Nutrient breakdown** targeting specific values for breakfast, lunch, and dinner.
- **Recipe recommendations** using an optimization algorithm to match predicted nutritional targets.
- **Fast Inference**: All PyTorch models and scalers are loaded once into memory upon server startup.
- **Flutter Ready**: Fully configured CORS to allow requests from any frontend (`*`).

## Project Structure

This repository is strictly organized into inference code (`api/`), static models (`models/`), and training utilities (`training/`).

```
.
├── api/                  # FastAPI inference endpoints & schemas
│   ├── main.py           # App entry point & lifespan
│   ├── routers/          # API endpoints (e.g., /generate-plan)
│   ├── services/         # Model loading and inference logic
│   └── ...
├── models/               # Pre-trained artifacts (PTH & PKL)
│   ├── encoders/         
│   ├── scalers/          
│   └── ...               # Neural network state dicts
├── training/             # Scripts for retraining (not used by API)
│   ├── datasets/         # Raw CSVs for training
│   └── backend.py        # Training pipeline script
├── Dockerfile            # Lightweight Docker image
├── railway.json          # Deployment config
└── requirements.txt      # Production dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/meal-recommend-ai.git
   cd meal-recommend-ai
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Local Development

Start the API server locally on port `8000`:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

## API Documentation

### `POST /generate-plan`
Accepts a JSON payload containing user data and returns a customized diet plan.

**Example Request:**
```json
{
  "age": 25,
  "gender": "M",
  "weight": 75.0,
  "height": 1.78,
  "activity_level": "Active",
  "goal": "Lose Weight"
}
```

## Railway Deployment Instructions

This repository is pre-configured for one-click deployment to [Railway.app](https://railway.app/).

1. Push this repository to your GitHub account.
2. Log in to Railway and create a **New Project** -> **Deploy from GitHub repo**.
3. Select this repository.
4. Railway will automatically detect the `Dockerfile` and `railway.json`.
5. The platform automatically assigns the `$PORT` environment variable and routes traffic securely.

*(No further configuration is required. The API will be live within 2-3 minutes!)*
