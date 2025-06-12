# 💼 Salary Prediction Using Linear Regression

This project aims to predict salaries based on an individual's years of experience using a **Simple Linear Regression** model. It is a beginner-friendly machine learning project that demonstrates how to handle a dataset, perform EDA, train a regression model, evaluate its performance, and visualize the results.

---

## 📌 Table of Contents

- [📁 Project Overview](#-project-overview)
- [📊 Dataset Info](#-dataset-info)
- [📚 Tech Stack & Libraries](#-tech-stack--libraries)
- [🔍 Problem Statement](#-problem-statement)
- [💡 Proposed Solution](#-proposed-solution)
- [🧠 Model Implementation Steps](#-model-implementation-steps)
- [📈 Results](#-results)
- [📝 Conclusion](#-conclusion)
- [🚀 Future Scope](#-future-scope)
- [📎 References](#-references)
- [📷 Screenshots](#-screenshots)

---

## 📁 Project Overview

In this project, we developed a machine learning model using **Linear Regression** to predict employee salaries based on their experience. It shows how ML models can be useful for HR planning and salary estimation. The entire process includes data loading, cleaning, exploration, model training, evaluation, and visualization.

---

## 📊 Dataset Info

- **Source**: [Kaggle – Salary Data](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)
- **Records**: 30  
- **Features**:
  - `YearsExperience`: Numeric value showing how many years of experience an individual has.
  - `Salary`: The corresponding salary of the individual.

---

## 📚 Tech Stack & Libraries

- **Language**: Python 3.x
- **Libraries**:
  - `pandas` – for data manipulation
  - `numpy` – for numerical operations
  - `matplotlib` & `seaborn` – for data visualization
  - `scikit-learn` – for Linear Regression model and evaluation

---

## 🔍 Problem Statement

Companies often need to estimate employee salaries for planning, budgeting, and recruitment. Manually predicting salaries is subjective and inconsistent. The aim of this project is to use a simple machine learning model to **predict salary based on years of experience**. This ensures more **accurate and objective decision-making** in HR management.

---

## 💡 Proposed Solution

We built a supervised ML model using **Simple Linear Regression** to learn the linear relationship between experience and salary. The model was trained on a small but clean dataset and then evaluated using R² score and mean squared error.

---

## 🧠 Model Implementation Steps

1. **Data Loading**  
   Loaded the CSV dataset using Pandas.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized the relationship between experience and salary.
   - Used scatter plots and correlation coefficients.

3. **Data Splitting**  
   - Split the dataset into 80% training and 20% testing using `train_test_split`.

4. **Model Training**  
   - Used `LinearRegression()` from scikit-learn to fit the training data.

5. **Prediction & Evaluation**  
   - Used the model to predict salaries on the test set.
   - Evaluated with metrics: **Mean Squared Error (MSE)** and **R² Score**.

6. **Visualization**  
   - Plotted regression line over the scatter plot of actual data points.

---

## 📈 Results

- **Intercept**: ₹[Base Salary]  
- **Coefficient**: ₹[Salary increase per year]  
- **Mean Squared Error**: *Very Low*  
- **R² Score**: *Close to 1 (High Accuracy)*

The regression line closely fits the data, confirming that salary increases linearly with years of experience.

---

## 📝 Conclusion

This project demonstrates that even a simple linear model can yield meaningful predictions when applied to well-prepared data. The model can serve as a basic tool for HR departments to estimate compensation based on experience.

---

## 🚀 Future Scope

- Include more features such as job title, education level, location, and skillset.
- Experiment with **Polynomial Regression** or **Decision Tree Regressors**.
- Deploy as a web application using **Streamlit** or **Flask**.
- Automate real-time predictions for HR departments.

---

## 📎 References

- Dataset: [Kaggle - Salary Data](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)  
- [Scikit-learn Linear Regression Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Matplotlib Documentation](https://matplotlib.org/)  
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

## 📷 Screenshots

| EDA Visualization                     | Regression Line Plot                    |
|--------------------------------------|-----------------------------------------|
| ![scatter](screenshots/eda_plot.png) | ![line](screenshots/regression_line.png)|

---

## 📬 Connect With Me

- **Name**: Divyanjali Gopisetty  
- **College**: Newton School of Technology – ADYPU  
- **Specialization**: B.Tech CSE (AI & ML)  
- **Email**: [your.email@example.com]  
- **LinkedIn**: [Your LinkedIn URL]  
