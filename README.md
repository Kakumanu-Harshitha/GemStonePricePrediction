## GemStone Price Prediction

An end-to-end Machine Learning project that predicts gemstone/diamond prices using regression models.
This project includes data ingestion, preprocessing, model training, evaluation, artifact saving, and a web interface for real-time predictions.

## 🚀 Key Features

Automated data ingestion from a GitHub raw dataset

Data preprocessing using pipelines (scaling + ordinal encoding)

Model training with Linear Regression, Lasso, Ridge, ElasticNet

Model selection based on best R² score

## Saves artifacts:

raw.csv, train.csv, test.csv

preprocessor.pkl

model.pkl

Prediction pipeline with support for web deployment

FastAPI or Flask interface for user input and prediction

MLflow integration for experiment tracking

Detailed logging and custom exception handling

###📂 Project Structure
GemStonePricePrediction/
│
├── main.py

├── src/

│   └── DiamondPricePrediction/

│       ├── components/

│       │   ├── data_ingestion.py

│       │   ├── data_transformation.py

│       │   ├── model_trainer.py

│       │   └── model_evaluation.py

│       ├── pipelines/

│       │   ├── training_pipeline.py

│       │   └── prediction_pipeline.py

│       ├── utils/

│       │   └── utils.py

│       ├── logger/

│       │   └── logging.py

│       ├── exception.py

│       └── __init__.py

├── templates/        
                                             
├── artifacts/            
                                                
├── logs/  
                        
├── requirements.txt

└── README.md

⚙️ Installation
1. Clone the repository
```bash
git clone https://github.com/Kakumanu-Harshitha/GemStonePricePrediction.git
cd GemStonePricePrediction
```
2. Create and activate a virtual environment
```bash
python -m venv gemStone
.\gemStone\Scripts\activate      # Windows
# source gemStone/bin/activate   # macOS/Linux
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

📊 Training the Model

Run the training pipeline:
```bash
python -m src.DiamondPricePrediction.pipelines.training_pipeline
```

This will:

Download the dataset

Perform train-test split

Transform data

Train and evaluate multiple models

Save the best model to artifacts/model.pkl

Save preprocessing pipeline to artifacts/preprocessor.pkl

## 🔮 Running Predictions
Option 1: Flask
```bash
python main.py

```

**Option 2: FastAPI**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```


Example JSON request:

{
  "carat": 0.5,

  "depth": 61,

  "table": 58,

  "x": 5.1,

  "y": 5.2,

  "z": 3.1,

  "cut": "Ideal",

  "color": "E",

  "clarity": "SI1"
}

But in this project I used FastApi

# 🧮 Model Evaluation

Evaluation includes:

RMSE

MAE

R² Score

All metrics are logged through MLflow (if enabled).

# 📝 Logging

Logs are stored in the logs/ directory with timestamped filenames:

logs\12_11_2025_20_58_36.log

# 📦 Artifacts

Training generates:

artifacts/raw.csv

artifacts/train.csv

artifacts/test.csv

artifacts/preprocessor.pkl

artifacts/model.pkl

These are automatically used during prediction.

# 🐞 Exception Handling

A custom exception class provides:

File name of error

Line number

Error message
Useful for debugging during model development and API failures.

## 📌 Requirements

Key libraries:

pandas

numpy

scikit-learn

FastAPI / Flask

MLflow

Python 3.8+

Full list in requirements.txt.

## 🚀 Future Improvements

Add XGBoost / RandomForest models

Hyperparameter tuning (GridSearchCV)

Deployment using Docker

Cloud deployment (AWS, Azure, GCP)

## 📬 OWNER

[Harshitha Kakumanu](https://github.com/Kakumanu-Harshitha)

