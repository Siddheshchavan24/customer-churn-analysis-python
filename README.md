# 📊 Customer Churn Analysis with Machine Learning
This project predicts whether a customer will churn (leave) or stay with a telecom service provider using data analysis and machine learning. It demonstrates how businesses can use data-driven insights to improve customer retention.

## 📁 Dataset
The dataset used is [`Telco-Customer-Churn.csv`](Telco-Customer-Churn.csv), which includes customer demographics, account information, and services used.

## 🧠 Project Overview
### ✅ Step 1: Data Cleaning & Preprocessing
- Loaded the **Telco Customer Churn dataset**
- Handled missing values and dropped irrelevant columns
- Encoded categorical features using one-hot encoding

### 📊 Step 2: Exploratory Data Analysis (EDA)
- Visualized customer churn by contract type
- Created a correlation heatmap to identify important relationships
- Extracted key insights from the data

### 🧪 Step 3: Model Building
- Trained a **Random Forest Classifier** for churn prediction
- Evaluated with:
  - Accuracy
  - Confusion Matrix
  - Classification Report

### 💡 Step 4: Streamlit Web App (Optional)
- Built a simple app (`app.py`) to predict churn
- Takes user input and shows churn likelihood
- Example output:  
  ✅ This customer is not likely to churn (Probability: 0.29)

## 📊 Visualizations

### Churn by Contract Type
This chart shows that customers with month-to-month contracts are far more likely to churn than those with long-term contracts.

![Churn by Contract Type](visuals/churn_by_contract.png)

### Correlation Heatmap (Numeric Features)
This heatmap shows the correlation between numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges`.

![Correlation Heatmap](visuals/correlation_matrix.png)

### Correlation Heatmap (All Encoded Features)
Here’s a full-feature correlation heatmap including binary and categorical encoded features.

![Full Correlation Matrix](visuals/correlation_matrix_viz.png)

## ✅ Model Highlights
- **Model Used:** Random Forest Classifier
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Churn Prediction Interface:** A Streamlit app to interactively predict churn


## 🚀 How to Run the Project
1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the notebook to understand the analysis:  
   `jupyter notebook churn_analysis.ipynb`
4. Launch the Streamlit app:  
   `streamlit run app.py`

   
## 🧠 Learnings
- Data cleaning and handling categorical variables
- Correlation analysis and feature selection
- Building a classification model
- Deploying a model with Streamlit

## ✨ Contributing
Feel free to fork the repo and submit a pull request with improvements!

## 📬 Contact
Connect with me on [LinkedIn](www.linkedin.com/in/siddheshchavan24)  
GitHub: [@YourGitHubUsername](https://github.com/Siddheshchavan24)

