#AI/ ML Professional's Salary Prediction#

This project showcases a production-grade machine learning system built following MLOps best practices, covering the full lifecycle from data preparation to cloud deployment.

```mermaid
flowchart TD
    A[Project Overview] 
    
    A --> B[Data Collection & Preparation]
    B --> B1[Kaggle Dataset]
    B1 --> B2[Salary Adjustment to 2025<br/>Using Inflation Rates]
    B2 --> B3[EDA & Train/Validation/Test Split]
    
    B --> C[Feature Engineering & Preprocessing]
    C --> C1[Categorical Encoding<br/>ColumnTransformer]
    
    C --> D[Model Development]
    D --> D1[Model Training<br/>Linear, SVM, Ensemble Models]
    D1 --> D2[Hyperparameter Tuning<br/>GridSearchCV]
    
    D --> E[Model Evaluation & Tracking]
    E --> E1[Evaluation on Test Set]
    E1 --> E2[Experiment Tracking<br/>MLflow]
    E2 --> E3[Best Model Selection]
    
    E --> F[Application Development]
    F --> F1[Flask REST API]
    F1 --> F2[Web UI<br/>HTML & CSS]
    
    F --> G[Deployment & CI/CD]
    G --> G1[Docker Containerization]
    G1 --> G2[GitHub Actions CI/CD]
    G2 --> G3[Azure Container Registry]
    G3 --> G4[Azure Web App Deployment]
    
    G --> H[Continuous Deployment]

```

##Data Collection & Preparation##

The dataset was sourced from Kaggle. While the data required minimal cleaning, the target variable (salary) spanned multiple years (2019–2023). To maintain real-world relevance, salary values were adjusted to 2025 equivalents using global and U.S. annual inflation rates.

Exploratory data analysis (EDA) was conducted to understand feature distributions and relationships. The dataset was then split into training, validation, and test sets. Categorical variables were processed using Scikit-learn’s ColumnTransformer, ensuring consistent preprocessing across all models.

##Model Training & Hyperparameter Optimization##

A diverse set of supervised learning algorithms was trained and evaluated, including:

- Linear models: Ridge, Lasso, Elastic Net

- Support Vector Machine (SVM)

- Ensemble methods: Random Forest, AdaBoost, Gradient Boosting

Hyperparameters were optimized using GridSearchCV, enabling systematic and fair model comparison.

##Model Evaluation & Experiment Tracking##

All models were evaluated using a held-out test dataset. Experiments—including models, parameters, and performance metrics—were tracked using MLflow, ensuring reproducibility and transparency. The best-performing model was selected for production deployment.

##Application & User Interface##

The trained model was exposed via a Flask-based REST API. A user-friendly web interface was developed using HTML and CSS, allowing end users to interact with the model through a clean and intuitive UI.

##Deployment & CI/CD##

The application was containerized using Docker and deployed via an automated CI/CD pipeline with GitHub Actions. Docker images were pushed to Azure Container Registry, and the application was deployed on Azure Web App. Continuous deployment was enabled to support rapid iteration and future updates.

##Key Skills Demonstrated##

- End-to-end machine learning pipeline design

- Feature engineering and preprocessing with Scikit-learn

- Model selection and hyperparameter tuning

- Experiment tracking with MLflow

- REST API development with Flask

- Containerization with Docker

- CI/CD automation using GitHub Actions

- Cloud deployment on Microsoft Azure

Excited to experience this cool app? Click [Maban](https://ai-salary-prediction-b4ayfph0f5buekgq.australiaeast-01.azurewebsites.net/)
