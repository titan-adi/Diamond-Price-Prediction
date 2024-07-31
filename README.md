# Diamond-Price-Prediction
Diamond Price Prediction
This project aims to predict the prices of diamonds using various features. The dataset contains information about the price and attributes of nearly 54,000 diamonds, including the carat weight, cut, color, clarity, and dimensions.

Table of Contents
Overview
Dataset
Features
Installation
Usage
Modeling and Evaluation
Results
Contributing
License
Overview
The goal of this project is to build a machine learning model that can accurately predict the price of a diamond based on its characteristics. The project includes data preprocessing, exploratory data analysis, feature engineering, and the implementation of various regression models.

Dataset
The dataset used in this project consists of diamonds with the following attributes:

price: Price in US dollars (target variable)
carat: Carat weight of the diamond
cut: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
color: Diamond color, ranging from J (worst) to D (best)
clarity: Diamond clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
depth: Total depth percentage = z / mean(x, y) = 2 * z / (x + y)
table: Width of the top of the diamond relative to the widest point
x: Length in mm
y: Width in mm
z: Depth in mm
Features
Numeric Features
carat
depth
table
x
y
z
Categorical Features
cut
color
clarity
Installation
To run this project, you need to have Python and the following packages installed:

numpy
pandas
seaborn
matplotlib
scikit-learn
xgboost
You can install these packages using pip:

bash
Copy code
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
Usage
To use this project:

Clone the repository and navigate to the project directory.
Load the dataset and preprocess it by following the steps in the code.
Train the model using the provided scripts and pipelines.
Evaluate the model's performance on the test set.
python
Copy code
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('diamonds.csv')

# Data preprocessing
# ... (preprocessing steps)

# Model training and evaluation
# ... (model training and evaluation steps)

# Results
# ... (evaluation results)
Modeling and Evaluation
The following models were built and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
K-Nearest Neighbors Regressor
XGBoost Regressor
The models were evaluated using 10-fold cross-validation with the negative root mean square error (RMSE) as the scoring metric.

Results
The XGBoost Regressor model was found to be the best performing model with the following evaluation metrics on the test set:

R²: 0.98
Adjusted R²: 0.98
MAE: 224.75
MSE: 119,403.62
RMSE: 345.58
These metrics indicate a high level of accuracy in predicting diamond prices.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
