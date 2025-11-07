Automated Machine Learning (AutoML) Framework

This Streamlit application provides a robust, end-to-end framework for automating the machine learning workflow, from exploratory data analysis (EDA) and preprocessing to model benchmarking, hyperparameter tuning, and prediction. It is designed to quickly analyze CSV datasets and deliver actionable model results for classification, regression, or clustering tasks.

✨ Features

Data Upload: Simple CSV file upload interface.

Automated EDA: Generates a comprehensive, interactive Exploratory Data Analysis report using Sweetviz.

Automated Preprocessing (Classification/Regression):

Handles missing values (mode for categorical, mean for numerical).

Removes outliers using the Interquartile Range (IQR) method.

Applies Label Encoding and standard scaling.

Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance classes in classification tasks.

Task Detection: Automatically detects if the task is Classification, Regression, or Clustering based on the selected target column.

Model Benchmarking: Trains and compares multiple standard ML models (Random Forest, Logistic Regression, SVR, etc.) and selects the best performer.

Hyperparameter Tuning: Optional automated tuning (via GridSearchCV or RandomizedSearchCV) on the best model.

Clustering Workflow (Clustering): Visualizes the optimal number of clusters using the Elbow Method and validates results with the Silhouette Score and PCA visualization.

Model Persistence: Allows downloading the trained, best-performing model as a .pkl file.

Prediction & Summary: Provides a form for testing predictions and uses a HuggingFace transformer pipeline to generate a textual summary of the model's performance and prediction.

🚀 Setup and Installation

Prerequisites

Python 3.8+

pip (Python package installer)

Step-by-Step Installation

Clone the Repository (or save app.py locally):

git clone <your-repository-url>
cd <your-repository-name> # or the directory containing app.py


Create and Activate a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows


Install Dependencies:

This project relies on standard data science libraries, Streamlit, Sweetviz for EDA, imblearn for SMOTE, and transformers for the summary generation.

pip install streamlit pandas numpy scikit-learn matplotlib seaborn sweetviz imblearn transformers


⚙️ Usage

Run the Streamlit App:

Start the application from your terminal:

streamlit run app.py


Access the Application:

The application will automatically open in your web browser, typically at http://localhost:8501.

Workflow:

Upload Data: Upload your CSV file using the file uploader.

Review EDA: The interactive Sweetviz report will appear for detailed data exploration.

Select Target: Choose the column you want to predict in the "Select Target Column" section.

Run Benchmarking: The framework will automatically run preprocessing, benchmark models, and select the best one.

Tune (Optional): Select "Yes" to apply automated hyperparameter tuning.

Predict: Use the input fields under "Test Prediction and Summary" to see a live prediction and a written summary of the entire process.

🛠️ Key Libraries Used

Library

Purpose

Streamlit

Web application framework

pandas / numpy

Data manipulation and numerical operations

scikit-learn

Core ML algorithms, preprocessing, and metrics

sweetviz

Automated Exploratory Data Analysis (EDA)

imblearn

Handling class imbalance (SMOTE)

transformers

Generating natural language summaries

matplotlib / seaborn

Data visualization for clustering and metrics
