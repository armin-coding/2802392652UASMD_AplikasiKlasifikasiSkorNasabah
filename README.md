# Customer Credit Score Classification

A machine learning web application that predicts customer credit score categories based on financial history, credit usage, and payment behaviour. Built with Python and Streamlit, this project demonstrates how a trained classification model can be deployed through an interactive interface for credit-related analysis.

## Overview

Credit score classification helps categorize customers according to their financial and credit-related characteristics. This project uses a trained machine learning model to predict a customer's credit score category from selected personal, income, debt, and payment information.

The application integrates a preprocessing pipeline and a trained classification model to transform user inputs into a prediction.

The model returns one of three categories:

- **Good** — a good credit score classification.
- **Standard** — a standard credit score classification.
- **Poor** — a poor credit score classification.

These categories represent the model's predictions, not official credit bureau assessments.

## Features

- **Interactive web interface** built with Streamlit.
- **Credit score classification** into Good, Standard, or Poor.
- **Financial data inputs** covering income, debt, credit history, and payment behaviour.
- **Automated preprocessing** using a saved preprocessing pipeline.
- **Reusable model artifacts** loaded from serialized files.
- **Separated inference logic** for better organization and maintainability.
- **Cached model loading** to avoid repeatedly initializing the model during application interactions.

## Tech Stack

- **Python 3.11** — application development
- **Pandas** — structured input data handling
- **Scikit-learn** — machine learning preprocessing and prediction
- **Joblib** — loading serialized model artifacts
- **Pickle** — loading model metadata
- **Streamlit** — interactive web application

## Input Features

The application collects 11 input features across three groups.

### 1. Personal and Employment Information

| Feature | Description |
|---|---|
| Age | Customer's age |
| Occupation | Customer's occupation |
| Annual Income | Customer's annual income |

### 2. Credit and Loan History

| Feature | Description |
|---|---|
| Number of Loans | Total number of loans |
| Outstanding Debt | Current outstanding debt |
| Changed Credit Limit | Change in credit limit |
| Credit History Age | Length of credit history, entered in months |
| Amount Invested Monthly | Monthly investment amount |
| Number of Delayed Payments | Number of delayed payments |

### 3. Payment Behaviour

| Feature | Description |
|---|---|
| Credit Mix | Categorized as Good, Standard, or Bad |
| Payment Behaviour | Spending and payment-pattern category |

The input values are passed to the inference pipeline, which aligns the data with the feature structure expected by the trained model.

## Project Structure

```text
2802392652UASMD_AplikasiKlasifikasiSkorNasabah/
├── files/                    # Supporting project files
├── models/
│   ├── best_model.pkl        # Trained classification model
│   ├── preprocessor.pkl      # Saved preprocessing pipeline
│   └── model_metadata.pkl    # Feature and column metadata
├── app.py                   # Streamlit user interface
├── inferencing.py            # Model loading and prediction logic
└── requirements.txt          # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.11
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/armin-coding/2802392652UASMD_AplikasiKlasifikasiSkorNasabah.git
```

### 2. Navigate to the Project Directory

```bash
cd 2802392652UASMD_AplikasiKlasifikasiSkorNasabah
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit, typically:

```text
http://localhost:8501
```

## How It Works

The application follows a machine learning inference workflow.

1. **User input:** The customer enters financial and credit-related information through the Streamlit interface.
2. **Data structuring:** The application organizes the submitted values into a dictionary and converts them into a Pandas DataFrame.
3. **Feature alignment:** The inference module uses saved metadata to identify the expected numerical and categorical columns and arrange them in the correct order.
4. **Preprocessing:** The saved preprocessing pipeline transforms the input data into the representation expected by the trained model.
5. **Classification:** The trained model predicts the customer's credit score category.
6. **Result display:** The application displays the predicted category through the Streamlit interface.

### Architecture

```text
User Input
    |
    v
Streamlit Interface (app.py)
    |
    v
ModelInferencer (inferencing.py)
    |
    v
Feature Alignment and Preprocessing
    |
    v
Trained Classification Model
    |
    v
Good / Standard / Poor
```

## Model Evaluation

The application uses a serialized trained model, `best_model.pkl`, for prediction.

To provide a complete evaluation of the model, the following metrics should be documented using results from an appropriate test dataset:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For a multiclass classification problem, precision, recall, and F1-score should ideally be reported for each class, alongside an appropriate average such as macro F1-score.

**Note:** This README does not claim a specific accuracy or identify the training algorithm because those details should be supported by the model training and evaluation results.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- The model may not generalize equally well across different customer populations.
- A predicted category is not equivalent to an official credit score or lending decision.
- The application does not establish whether a customer should be approved for a loan.
- The serialized model and preprocessing artifacts must remain compatible with the expected input features.

## Future Improvements

- Add classification evaluation metrics and a confusion matrix.
- Include visualizations of credit-related input characteristics.
- Improve input validation and error handling.
- Add explanations of the factors influencing individual predictions.
- Document the training dataset and model selection process.
- Add automated tests for the inference pipeline.
- Deploy the application to a public hosting platform.

## Disclaimer

This project is intended for educational and demonstration purposes. Its predictions are generated by a machine learning model and should not be used as the sole basis for credit approval, loan eligibility, or other consequential financial decisions.

## Author

**GitHub Repository:** [2802392652UASMD_AplikasiKlasifikasiSkorNasabah](https://github.com/armin-coding/2802392652UASMD_AplikasiKlasifikasiSkorNasabah)
