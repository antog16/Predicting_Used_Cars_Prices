# Predicting Used Car Prices

This repository contains a machine learning model for predicting the prices of used cars based on a variety of features. The project aims to demonstrate the process of data cleaning, exploratory data analysis (EDA), feature engineering, and building a predictive model using various machine learning algorithms (RandomForestRegressor & XGBoostRegressor).

An app on streamlit was created to deploy the model and generate predictions: https://predictingusedcarsprices.streamlit.app/

## Project Overview

The goal of this project is to predict the prices of used cars from a dataset containing information about different car attributes. The dataset includes features like car make, model, year, mileage, fuel type, and others that affect the price of a used car.

## Technologies Used

- **Python**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Regex**: Regular Expressions for string pattern matching and data cleaning
- **XGBoost**: Gradient Boosting library for predictive modeling
- **Matplotlib** & **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and utilities
- **Jupyter Notebook**: Interactive development and analysis

## Dataset

The dataset used in this project can be found [here](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data). It includes multiple columns such as:

- Brand & Model: Identify the brand or company name along with the specific model of each vehicle.
- Model Year: Discover the manufacturing year of the vehicles, crucial for assessing depreciation and technology advancements.
- Mileage: Obtain the mileage of each vehicle, a key indicator of wear and tear and potential maintenance requirements.
- Fuel Type: Learn about the type of fuel the vehicles run on, whether it's gasoline, diesel, electric, or hybrid.
- Engine Type: Understand the engine specifications, shedding light on performance and efficiency.
- Transmission: Determine the transmission type, whether automatic, manual, or another variant.
- Exterior & Interior Colors: Explore the aesthetic aspects of the vehicles, including exterior and interior color options.
- Accident History: Discover whether a vehicle has a prior history of accidents or damage, crucial for informed decision-making.
- Clean Title: Evaluate the availability of a clean title, which can impact the vehicle's resale value and legal status.
- Price: Access the listed prices for each vehicle, aiding in price comparison and budgeting.

The dataset is cleaned and preprocessed for analysis and model training.

## Key Features

- **Data Cleaning**: Handling missing values, outliers, and categorical variables.
- **Feature Engineering**: Creating new features to improve the model's predictive power.
- **Model Building and Evaluation**: Training multiple machine learning models and comparing their performance.

## Conclusion

This project demonstrates how to predict the prices of used cars based on various features. The goal is to showcase the process of data cleaning, model selection, and evaluation. The results of the project can be extended to more advanced models and optimizations to improve prediction accuracy.

Let me know if you need any adjustments!
