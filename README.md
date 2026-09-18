# Vehicular CO₂ Emission Prediction

A data-driven machine learning project for predicting **vehicle CO₂
emissions (g/km)** from vehicle specifications and fuel-consumption
characteristics. The project covers the complete workflow from data
preprocessing and model development to experiment management and
deployment through a **FastAPI web application**.

> **Internship Project:** National Institute of Advanced Manufacturing
> Technology (NIAMT), Ranchi\
> **Author:** Animesh Porwal\
> **Duration:** June 2026 -- July 2026

------------------------------------------------------------------------

## Overview

Vehicular transportation is a major source of carbon dioxide emissions.
Conventional emission measurement generally requires controlled testing
environments and specialized equipment. This project explores a
machine-learning-based alternative that estimates CO₂ emissions using
readily available vehicle information.

The project follows an end-to-end ML workflow:

``` text
Dataset
   ↓
Data Cleaning
   ↓
Data Preparation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
FastAPI Deployment
   ↓
Web Interface
   ↓
CO₂ Emission Prediction
```

The final application allows a user to enter vehicle information and
receive a predicted CO₂ emission value along with an
environmental-impact category.

------------------------------------------------------------------------

## Objectives

-   Clean and preprocess the vehicle CO₂ emission dataset.
-   Analyze relationships between vehicle characteristics, fuel
    consumption, and CO₂ emissions.
-   Train multiple regression models.
-   Compare model performance using standard regression metrics.
-   Optimize the selected model through hyperparameter tuning.
-   Save the trained model and preprocessing pipeline for reuse.
-   Deploy the prediction system using FastAPI.
-   Present predictions through an easy-to-use HTML interface.
-   Categorize predicted emissions to make the result easier to
    interpret.

------------------------------------------------------------------------

## Dataset

The project uses the public **CO₂ Emissions by Vehicles** dataset from
Kaggle.

The dataset contains vehicle specifications, fuel-consumption
information, and the corresponding CO₂ emissions in grams per kilometer.

### Main features

-   Make
-   Model
-   Vehicle Class
-   Engine Size (L)
-   Cylinders
-   Transmission
-   Fuel Type
-   Fuel Consumption City (L/100 km)
-   Fuel Consumption Hwy (L/100 km)
-   Fuel Consumption Comb (L/100 km)
-   Fuel Consumption Comb (mpg)

### Target

-   **CO₂ Emissions (g/km)**

Dataset source:

https://www.kaggle.com/datasets/debajyotipodder/CO2-emission-by-vehicles

------------------------------------------------------------------------

## Machine Learning Workflow

### 1. Data Preprocessing

The preprocessing pipeline includes:

-   Removal of duplicate records.
-   Transformation of categorical variables into numerical
    representations.
-   Scaling of numerical features.
-   Train/test splitting.
-   Saving the preprocessing pipeline so that the same transformations
    can be reused during inference.

The project uses Scikit-learn preprocessing components to keep the
transformation process consistent between training and deployment.

### 2. Feature Engineering

Categorical and numerical features are transformed into a format
suitable for machine-learning models.

The prediction pipeline uses vehicle specifications and fuel-consumption
characteristics as model inputs.

### 3. Model Development

Three regression models were evaluated:

-   XGBoost Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor

Hyperparameter optimization was performed to improve the performance of
the selected model.

------------------------------------------------------------------------

## Model Performance

The internship report compares the models using training and testing
RMSE and MAE.

  Model                 Train RMSE   Train MAE   Test RMSE    Test MAE
  ------------------- ------------ ----------- ----------- -----------
  XGBoost                    1.500       1.112       4.442       2.686
  Random Forest              1.776       1.113       4.224       2.393
  Gradient Boosting          2.045       1.683   **3.675**   **2.328**

The **Gradient Boosting Regressor** was selected as the final model
based on the reported test performance.

The evaluation used:

-   R² Score
-   Mean Absolute Error (MAE)
-   Mean Squared Error (MSE)
-   Root Mean Squared Error (RMSE)

------------------------------------------------------------------------

## Deployment

The trained model is deployed using **FastAPI**.

The application uses:

-   **FastAPI** --- backend/API framework
-   **Jinja2** --- HTML template rendering
-   **HTML/CSS** --- user interface
-   **Joblib** --- model and preprocessing-pipeline serialization
-   **Uvicorn** --- ASGI server

The application loads the saved preprocessing pipeline and trained
model, accepts vehicle information, performs the required
transformations, and returns the predicted CO₂ emission.

------------------------------------------------------------------------

## Project Structure

The repository is organized as an end-to-end machine-learning project:

``` text
CO2-Emission/
│
├── app/
│   ├── routers/
│   │   └── prediction.py
│   ├── main.py
│   └── predictor.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│
├── models/
│   ├── model.joblib
│   └── transformer.joblib
│
├── notebooks/
│
├── references/
│
├── reports/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── static/
│   └── ...
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── .dvc/
├── .venv/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── setup.py
├── test_environment.py
├── .gitignore
├── .dvcignore
├── LICENSE
└── README.md
```

> Some generated files, cached files, datasets, and model artifacts may
> not be committed to Git. DVC is used for dataset/pipeline management
> where configured.

------------------------------------------------------------------------

## Running the Project Locally

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd CO2-Emission
```

### 2. Create a virtual environment

On Ubuntu/Linux:

``` bash
python3 -m venv .venv
```

### 3. Activate the environment

``` bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### 4. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

If the FastAPI HTML templates are used and Jinja2 is not already
present:

``` bash
python -m pip install jinja2
```

### 5. Run the FastAPI application

From the project root:

``` bash
python -m uvicorn app.main:app --reload
```

The application will be available at:

``` text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## FastAPI Application

The main application is defined in:

``` text
app/main.py
```

The prediction endpoint is handled by:

``` text
app/routers/prediction.py
```

Prediction/model loading logic is contained in:

``` text
app/predictor.py
```

The application uses Jinja2 templates to render the web pages:

``` text
templates/
├── index.html
└── result.html
```

Static assets such as CSS are served from:

``` text
static/
```

------------------------------------------------------------------------

## Model and Preprocessing Artifacts

The deployment application uses serialized artifacts generated during
model development:

``` text
models/model.joblib
models/transformer.joblib
```

`transformer.joblib` stores the preprocessing pipeline, while
`model.joblib` stores the trained regression model.

Keeping the preprocessing pipeline together with the deployed model
helps ensure that inference data receives the same transformations used
during training.

------------------------------------------------------------------------

## MLOps and Reproducibility

The project incorporates MLOps practices to make the workflow more
organized and reproducible.

### DVC

**DVC (Data Version Control)** is used for dataset and ML-pipeline
management.

The project includes:

``` text
dvc.yaml
dvc.lock
.dvc/
```

The pipeline is divided into stages such as:

-   Data cleaning
-   Data preparation
-   Feature engineering
-   Model training

Pipeline parameters are maintained in:

``` text
params.yaml
```

### Git and GitHub

Git is used for source-code version control and GitHub is used for
repository hosting.

### MLflow

MLflow is used for experiment tracking.

------------------------------------------------------------------------

## Technologies Used

### Programming

-   Python 3.14

### Data Science / Machine Learning

-   Pandas
-   NumPy
-   Scikit-learn
-   SciPy
-   Matplotlib
-   Seaborn
-   Joblib

### Machine Learning Models

-   XGBoost
-   Random Forest
-   Gradient Boosting

### Web Development

-   FastAPI
-   Uvicorn
-   Jinja2
-   HTML
-   CSS

### MLOps / Development Tools

-   DVC
-   MLflow
-   Git
-   GitHub
-   VS Code
-   Jupyter Notebook

------------------------------------------------------------------------

## Results

The final Gradient Boosting model achieved the lowest reported test
error among the evaluated models:

``` text
Test RMSE: 3.675
Test MAE : 2.328
```

The project also includes an actual-vs-predicted CO₂ emission
visualization. The plotted predictions follow the recorded values
closely, with deviations that are expected when working with real-world
data.

------------------------------------------------------------------------

## Limitations

The current system has several limitations:

-   It is based on a single publicly available vehicle-emission dataset.
-   Newer vehicle models and geographical variations may not be fully
    represented.
-   Real-world factors such as driving behaviour, traffic conditions,
    road gradients, vehicle age, and maintenance are not included.
-   The system predicts CO₂ emissions only and does not currently cover
    pollutants such as NOx, CO, or particulate matter.
-   There is a difference between training and testing errors, leaving
    room for further improvement in generalization.

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   Training on larger and more diverse datasets.
-   Adding newer vehicle models and regional data.
-   Incorporating vehicle age and maintenance information.
-   Including driving behaviour and traffic conditions.
-   Adding environmental factors.
-   Exploring deep-learning and hybrid ensemble approaches.
-   Extending prediction to NOx, CO, and particulate matter.
-   Deploying the application to the cloud.
-   Integrating the prediction system with automotive platforms for
    real-time estimation.

------------------------------------------------------------------------

## Internship Context

This project was developed as part of an academic internship at:

**National Institute of Advanced Manufacturing Technology (NIAMT),
Ranchi**

under the guidance of:

**Dr. Nidhi Kumari**\
Assistant Professor\
Department of Electronics and Computer Engineering\
NIAMT Ranchi

The project focused on applying machine learning and MLOps principles to
a practical environmental prediction problem.

------------------------------------------------------------------------

## Author

**Animesh Porwal**

B.Tech --- Computer Science and Design\
Madhav Institute of Technology and Science, Gwalior

GitHub: https://github.com/anni0955\
Portfolio: https://anni0955.github.io

------------------------------------------------------------------------

## References

1.  D. Podder, *CO2 Emission by Vehicles Dataset*, Kaggle, 2023.
2.  F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*,
    Journal of Machine Learning Research, 2011.
3.  A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras &
    TensorFlow*, 3rd ed., O'Reilly Media, 2022.
4.  T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of
    Statistical Learning*, 2nd ed., Springer, 2009.
5.  C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer,
    2006.
6.  L. Breiman, *Random Forests*, Machine Learning, 2001.
7.  J. H. Friedman, *Greedy Function Approximation: A Gradient Boosting
    Machine*, The Annals of Statistics, 2001.
8.  T. Chen and C. Guestrin, *XGBoost: A Scalable Tree Boosting System*,
    KDD, 2016.
9.  W. McKinney, *Python for Data Analysis*, 3rd ed., O'Reilly Media,
    2022.
10. FastAPI Documentation: https://fastapi.tiangolo.com/
