# Airline Price Prediction API

A FastAPI service that predicts airline ticket prices from flight details using a trained scikit-learn model.

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The project includes the trained model and preprocessing files:

- `model.pkl`
- `preprocess.pkl`
- `si.pkl`
- `si_cat.pkl`

Keep scikit-learn at version `1.2.2` because the model files were created with that version.

Render uses the Python version declared in `runtime.txt`. Python 3.11.9 is pinned because scikit-learn 1.2.2 does not provide a compatible wheel for Python 3.14.

The included `render.yaml` also sets `PYTHON_VERSION=3.11.9` and starts the service with Uvicorn on Render.

## Run

```bash
python -m uvicorn main:app --reload
```

The API runs at <http://127.0.0.1:8000>.

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## Endpoints

- `GET /` returns an API welcome message.
- `GET /health` checks whether the API is running.
- `POST /predict` returns a predicted ticket price.

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "Indigo",
    "source": "Delhi",
    "destination": "Mumbai",
    "departure": "Morning",
    "stops": "zero",
    "arrival": "Afternoon",
    "class": "Economy",
    "duration": 2.5,
    "days_left": 10
  }'
```

The response has this format:

```json
{
  "predicted_price": 6992.82
}
```
