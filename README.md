# 🚢 Titanic Survival Prediction – Machine Learning Web App

This project is a end-to-end machine learning pipeline that predicts the survival of Titanic passengers. It includes data preprocessing, model training, evaluation, and deployment using **Streamlit** and **Gradio**.

## 📌 Project Objectives

- Build an accurate predictive model for Titanic passenger survival.
- Identify key features influencing survival (e.g., sex, age, class).
- Apply proper data cleaning, transformation, and visualization.
- Deploy an interactive web app for public use.

## 📊 Dataset

- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Rows:** 891
- **Target:** `Survived` (0 = No, 1 = Yes)
- **Key Features:** `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`

## ⚙️ Data Preparation

- Missing values filled (`Age`, `Fare` with mean; `Embarked` with mode)
- Label encoding for `Sex`, one-hot encoding for `Embarked`
- Fare transformed using `log1p`; features scaled using `StandardScaler`
- Class imbalance addressed using **SMOTE**

## 🤖 Models Used

- **Random Forest Classifier** (`n_estimators=100`)
- **Logistic Regression** (with `GridSearchCV` tuning)

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC

## 🌐 Deployment Links

- 🔗 **Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1OISzIOdfoMKAh9-CFsqor6fn_zMHS5b3)
- 🚀 **Streamlit App:** [Live App](https://titanicmuhannad.streamlit.app/)
- 💻 **GitHub Repository:** [GitHub Link](https://github.com/Muhannadam/Titanic)

## 🧰 Technology Stack

- Python (pandas, numpy, scikit-learn, seaborn, matplotlib, imblearn)
- Streamlit & Gradio for UI
- Google Colab for development
- GitHub for version control

## 📁 Project Files

- `app.py` – Streamlit UI
- `requirements.txt` – Required packages
- `scaler.pkl` – Scaler object
- `logistic_regression_model.pkl` – Logistic regression model
- `random_forest_model.pkl` – Random forest model

## 🚧 Challenges Faced

- Handling missing/outlier values
- Choosing impactful features
- Dealing with data imbalance

## ✅ Conclusion

This project demonstrates a full ML workflow and public deployment.

---

### 👥 Team Members

| Name                    | Student ID |
|-------------------------|------------|
| Mohammed Talal Mursi    | 2503652    |
| Ghaith Omar Alhumaidi   | 2503650    |
| Muhannad Almuntashiri   | 2503649    |