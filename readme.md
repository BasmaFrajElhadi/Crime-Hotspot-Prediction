# Crime Hotspot Prediction

This project focuses on predicting crime hotspots in Chicago for the next 24 hours by forecasting likely locations (latitude and longitude), types of crimes (e.g., theft, assault), and the expected number of crimes per hour. The system aims to help law enforcement agencies take proactive measures to prevent crimes and improve community safety.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Source](#data-source)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Latitude and Longitude Prediction](#latitude-and-longitude-prediction)
  - [Crime Type Prediction](#crime-type-prediction)
- [Evaluation](#evaluation)
- [Results and Insights](#results-and-insights)
- [License](#license)

## Project Overview

The project predicts crime hotspots by combining forecasts for crime locations, types, and frequencies. It includes:

- Exploratory Data Analysis (EDA) to understand crime patterns in Chicago.
- Data preprocessing and feature engineering.
- Machine learning models for predicting crime coordinates (latitude and longitude) and crime types.
- Evaluation of model performance.
- Visualizations to provide insights into crime patterns.

## Data Description

The dataset used in this project contains information on crimes in Chicago from January 1, 2022, to December 31, 2023. The dataset includes the following columns:

- `date`: The date and time of the crime.
- `primary_type`: The type of crime committed.
- `arrest`: Whether an arrest was made.
- `domestic`: Whether the crime was domestic in nature.
- `longitude`: The longitude of the crime scene.
- `latitude`: The latitude of the crime scene.
- `district`: The police district where the crime occurred.
- `location_description`: The description of the crime location.
- `community_area`: The community area where the crime occurred.

## Data Source

The dataset used in this project was sourced from the City of Chicago's open data portal. The original dataset, "Crimes - 2001 to Present," can be accessed [here](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data).

## Exploratory Data Analysis (EDA)

The EDA phase involved the following analyses:

1. **Crime Type Analysis**: Identified the most common types of crimes.
2. **Arrest Rate Analysis**: Examined the percentage of crimes resulting in arrests.
3. **Temporal Analysis**: Analyzed crime frequency by time of day, day of the week, and month.
4. **Geographical Analysis**: Identified locations and districts with the highest crime rates.

## Data Preprocessing

- Standardized column names for consistency.
- Removed unnecessary columns (`id`, `case_number`, `updated_on`, `fbi_code`).
- Handled missing values by removing rows with missing data.
- Checked and validated latitude and longitude values.
- Encoded categorical variables using `LabelEncoder`.
- Extracted time-related features (hour, day, month, day of the week) from the `date` column.
- Resampled the data using SMOTE to address class imbalance in crime types.

## Modeling

### Latitude and Longitude Prediction

A `MultiOutputRegressor` with `RandomForestRegressor` was used to predict the coordinates (latitude and longitude) of crime scenes based on time and crime type features.

### Crime Type Prediction

A `RandomForestClassifier` was used to predict the type of crime. The dataset was resampled using SMOTE to handle class imbalance.

## Evaluation

The models were evaluated using the following metrics:

- **Latitude and Longitude Prediction**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R2)

- **Crime Type Prediction**:
  - Accuracy
  - Precision, Recall, and F1-Score
  - Confusion Matrix

## Results and Insights

- **Crime Type Distribution:** Theft and property-related crimes were the most common, while violent crimes like battery and assault were also prevalent.

- **Arrest Analysis:** A significant percentage of crimes did not result in arrests, highlighting potential areas for law enforcement improvement.

- **Temporal Trends:** Crimes peaked during nighttime hours and on weekends, particularly Fridays and Saturdays.

- **Geographical Insights:** Certain districts and public spaces were identified as crime hotspots, providing actionable insights for targeted interventions.



Many thanks to [Epsilon AI](https://github.com/EPSILON-AI)
