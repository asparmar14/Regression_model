# Hands-On Machine Learning: California Housing Prices

This repository contains code exercises from Chapter 2 of the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems" by Aurélien Géron. The exercise focuses on predicting housing prices in California using various machine learning regression models.

## Description

This Jupyter notebook (`california_housing_prices.ipynb`) explores the California housing dataset, aiming to predict median house prices. It covers data analysis, preprocessing, model selection, evaluation, and fine-tuning using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Skills and Tools Utilized

1. **Pandas**
2. **NumPy**
3. **Matplotlib**
4. **Seaborn**
5. **scikit-learn**
6. **Machine Learning**
7. **Regression Models** (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
8. **Data Visualization**
9. **Feature Engineering**
10. **Data Preprocessing** (Imputation, Encoding Categorical Variables, Scaling)
11. **Cross Validation**
12. **Hyperparameter Tuning** (Randomized Search)

## Contents

### Exploratory Data Analysis
- Initial exploration of the dataset, including summary statistics and visualizations.

### Data Preprocessing
- Handling missing values (`SimpleImputer`), encoding categorical variables, and scaling features (`StandardScaler`).

### Model Selection
- Comparison of regression models such as Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

### Evaluation
- Utilizing metrics like mean squared error (MSE), root mean squared error (RMSE) to evaluate model performance.

### Fine-Tuning
- Using randomized search for hyperparameter tuning to optimize the Random Forest Regressor.

### Feature Importance
- Determining feature importance to understand variables affecting house prices the most.


## Dependencies

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Usage
1. Create a virtualenv
```
pip install virtualenv
py -m venv hands_on_ml
hands_on_ml\Scripts\activate
```
2. Install necessary libraries
```
pip install pandas numpy matplotlib seaborn
pip install -U scikit-learn
```

3. Clone the repository and navigate to its directory:
```
git clone https://github.com/asparmar14/Regression_model.git

cd Regression_model
```

4. Launch Jupyter Notebook:
jupyter notebook california_housing_prices.ipynb

5. Follow the instructions in the notebook to execute and interact with the analysis.

## Acknowledgement

Special thanks to Aurélien Géron for the exercise provided in his book, which serves as the foundation for this repository.

## Author

- Anshul Parmar

Feel free to explore the notebook and provide any feedback or suggestions!





