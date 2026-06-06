# House Price Prediction

A regression project that predicts median house prices using the California Housing dataset. The goal was to build a clean linear regression pipeline and understand how different neighborhood features drive property values.

## Dataset

The California Housing dataset is built directly into scikit-learn so no file download is needed. It contains around 20,000 records with block-level housing data collected from the 1990 California census. The target variable is the median house value for each block, measured in units of $100,000.

    from sklearn.datasets import fetch_california_housing

## Features Used

Five features were selected based on correlation analysis and domain relevance.

    MedInc      Median income of the block (strongest predictor)
    HouseAge    Median age of houses in the block
    AveRooms    Average number of rooms per household
    AveOccup    Average number of occupants per household
    Population  Total population of the block

## Steps Covered

Data is loaded and converted into a Pandas DataFrame. Basic cleaning is done by checking for nulls and removing extreme outliers beyond the 99th percentile for room count and occupancy columns.

EDA includes a price distribution histogram, scatter plots of income and house age against price, and a full correlation heatmap to identify which features matter most.

Features are scaled using StandardScaler before training. The data is split 80/20 for training and testing.

A Linear Regression model is trained and evaluated using four metrics. A plot of actual versus predicted values is generated along with residual plots to check for systematic prediction errors. Coefficients are visualized and interpreted in plain business terms.

## Results

    MAE   0.53   average prediction error of around $53,000
    RMSE  0.73   typical error of around $73,000
    R²    0.65   model explains 65% of the variation in house prices

Median income turned out to be the strongest single driver of house price by a significant margin.

## Requirements

    pip install pandas numpy matplotlib seaborn scikit-learn

## How to Run

    python project2_house_price_prediction.py

Eight plots are saved automatically to the working directory.
