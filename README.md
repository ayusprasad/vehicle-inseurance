# Vehicle Insurance Prediction API

[![CI](https://github.com/ayusprasad/vehicle-inseurance/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/vehicle-inseurance/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![AWS](https://img.shields.io/badge/Deployment-AWS-232F3E?logo=amazonaws&logoColor=white)

A FastAPI application for vehicle insurance predictions, model training status, and optional cloud-backed model artifacts.

## Capabilities

- Browser form and JSON API for insurance predictions.
- Health and model-training status endpoints.
- Modular ingestion, validation, preprocessing, training, and prediction code.
- Optional MongoDB metadata and S3 artifact integration through environment configuration.

## Architecture

~~~text
FastAPI app.py → prediction/training pipelines → scikit-learn artifacts
                          ├─ MongoDB (optional metadata)
                          └─ S3 (optional model storage)
~~~

## Local setup

### Prerequisites

- Python 3.11 or newer
- MongoDB and AWS credentials only if using those integrations
- A trained model artifact for prediction routes

### Install

~~~bash
git clone https://github.com/ayusprasad/vehicle-inseurance.git
cd vehicle-inseurance
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Activate the virtual environment before the install command: use source .venv/bin/activate on macOS/Linux, or .venv/Scripts/Activate.ps1 in PowerShell.

### Configure and run

Copy .env.example to .env and set only the values needed for your deployment. Never commit real credentials. Then start the API:

~~~bash
uvicorn app:app --host 127.0.0.1 --port 5000
~~~

Open http://127.0.0.1:5000/docs for the interactive API, or GET /api/health for a health check. Prediction requires trained artifacts; the health endpoint does not connect to external services.

## Quality checks

~~~bash
python -m pip install -r requirements-dev.txt
ruff check --select E4,E7,E9,F app.py tests
pytest -q
python -m compileall -q app.py src tests
~~~

GitHub Actions runs these checks and builds the Docker image on pushes and pull requests. The workflow does not connect to AWS or MongoDB.

## Security

MongoDB and AWS credentials are read from environment variables. The temporary connection scripts that contained a MongoDB URI have been removed from the current branch; rotate any credential that was committed before this cleanup because Git history retains old versions.
