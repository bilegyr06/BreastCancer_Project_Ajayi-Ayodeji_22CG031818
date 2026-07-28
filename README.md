# Breast Cancer Prediction App

This repository contains a lightweight web application for predicting whether a breast cancer sample is likely to be malignant or benign. The app uses a trained machine learning model served through a FastAPI backend and exposed through a simple HTML form.

## Problem

Breast cancer diagnosis often requires careful review of multiple quantitative measurements. This project provides a small, accessible way to demonstrate how a classification model can turn a set of medical feature values into a binary prediction for screening purposes.

## Solution

The application collects six commonly used tumor characteristics from the user through a web form, sends them to a FastAPI endpoint, and uses a persisted scikit-learn model to return a prediction. The model is trained offline using the Breast Cancer Wisconsin dataset and saved with joblib so it can be loaded at runtime.

## Features

- Simple web-based prediction form for entering tumor measurements
- FastAPI backend with a POST endpoint for inference
- Persisted trained model loaded from disk at startup
- Jinja-powered HTML template for rendering the user interface
- Training script that builds, evaluates, and saves the model

## Architecture

The project is organized as a minimal full-stack prototype:

- Frontend: a server-rendered HTML form using Jinja templates
- Backend: FastAPI application in app.py
- Machine learning: a trained logistic regression pipeline in model/model_building.py
- Model persistence: joblib serialization in model/breast_cancer_model.pkl
- Database: none
- External services: none

## Tech Stack

### Frontend
- HTML
- Jinja2 templates

### Backend
- Python
- FastAPI
- Uvicorn
- python-multipart

### Machine Learning
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib
- seaborn

### Deployment
- Render (the project notes reference a hosted URL on Render)

## Project Structure

- app.py: FastAPI entry point and prediction routes
- templates/index.html: web form and result rendering
- model/model_building.py: training pipeline, evaluation, and model serialization
- model/breast_cancer_model.pkl: trained model artifact
- requirements.txt: project dependencies
- BreastCancer_hosted_webGUI_link.txt: deployment and repository metadata

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bilegyr06/BreastCancer_Project_Ajayi-Ayodeji_22CG031818.git
   cd BreastCancer_Project_Ajayi-Ayodeji_22CG031818
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If you want to retrain the model from scratch, run:
   ```bash
   python model/model_building.py
   ```

   This script trains a preprocessing-and-classification pipeline and saves the model to model/breast_cancer_model.pkl.

## Running Locally

Start the development server with:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open the following URL in your browser:

```text
http://127.0.0.1:8000/
```

## Usage

1. Open the app in your browser.
2. Enter the six feature values requested by the form:
   - Mean Radius
   - Mean Texture
   - Mean Perimeter
   - Mean Area
   - Mean Smoothness
   - Mean Compactness
3. Submit the form.
4. The app will display a prediction of either malignant or benign.

Screenshots can be added to the repository to illustrate the form and prediction output.

## API

The application exposes the following routes:

- GET /: renders the prediction form
- GET /redirect: redirects to the root route
- POST /predict: accepts the six input features as form fields and returns a rendered HTML response with the prediction result

## Machine Learning Details

The model training pipeline is defined in model/model_building.py.

- Dataset: Breast Cancer Wisconsin dataset from scikit-learn
- Features used: six selected measurements from the dataset
- Preprocessing: StandardScaler inside a scikit-learn pipeline
- Model: LogisticRegression
- Evaluation: the training script computes accuracy, precision, recall, F1 score, and a classification report for the train and test sets
- Inference: the trained pipeline is reloaded from disk with joblib and used for prediction

## Future Improvements

Possible next steps for this project include:

- Adding input validation and clearer error handling for malformed values
- Expanding the model to use a broader set of features
- Adding model explainability so users can understand which inputs contributed most to the prediction
- Introducing automated tests for the backend and prediction flow
- Containerizing the application for easier deployment and reproducibility
- Replacing the template-based UI with a modern frontend framework if the project grows

## Contributing

Contributions are welcome. If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes and test them locally
4. Open a pull request with a clear explanation of the update

## License

No license has been specified for this repository yet.
