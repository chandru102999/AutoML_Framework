import streamlit as st
import pandas as pd
import numpy as np
import sweetviz as sv
import tempfile
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from imblearn.over_sampling import SMOTE
from transformers import pipeline as hf_pipeline

st.title("🔐 Automatic ML Framework")

role = st.radio("Are you a Data Scientist?", ["Yes", "No"])

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("1️⃣ Preview of Uploaded Dataset")
    st.dataframe(df.head())

    st.subheader("2️⃣ Exploratory Data Analysis (Sweetviz)")
    report = sv.analyze(df)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        report.show_html(filepath=tmp_file.name, open_browser=False)
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=600, scrolling=True)

    if st.button("Download EDA Report as HTML"):
        with open(tmp_file.name, 'rb') as f:
            st.download_button("📥 Download EDA Report", f, file_name="EDA_Report.html")

    if role == "Yes":
        st.subheader("🧠 Custom Preprocessing")
        st.code("""
# Example: df['column'] = df['column'].fillna(df['column'].mean())
        """, language='python')
        user_code = st.text_area("Write your preprocessing code below:")
        if st.button("Run My Preprocessing"):
            exec(user_code, globals())
            st.success("✅ Custom preprocessing applied")
            st.dataframe(df.head())
    else:
        st.subheader("🔧 Automated Preprocessing")

        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        df = df.drop_duplicates()

        # Handle missing
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        # Encode rare labels
        for col in categorical_cols:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < 0.05].index
            df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        st.dataframe(df.head())

    st.subheader("3️⃣ Select Target Column")
    target_col = st.selectbox("Select target column (Leave empty for clustering)", ["None"] + df.columns.tolist())

    if target_col != "None":
        y = df[target_col]
        task_type = "classification" if y.nunique() <= 10 and y.dtype != float else "regression"
    else:
        task_type = "clustering"

    st.write(f"Detected Task: **{task_type.title()}**")

    if task_type == "clustering":
        numeric_df = df.select_dtypes(include=[np.number])

        st.subheader("🔢 Elbow Method")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(numeric_df)
            wcss.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        sns.lineplot(x=range(1, 11), y=wcss, ax=ax)
        ax.set_title("Elbow Method")
        ax.set_xlabel("Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        k = st.slider("Select clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(numeric_df)
        score = silhouette_score(numeric_df, df['Cluster'])
        st.success(f"✅ Silhouette Score: {score:.2f}")

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df)
        df['PC1'], df['PC2'] = pca_result[:, 0], pca_result[:, 1]
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax2)
        ax2.set_title("Clusters Visualized via PCA")
        st.pyplot(fig2)
        st.dataframe(df.head())
    else:
        if task_type == "classification":
            if df[target_col].value_counts(normalize=True).min() < 0.5:
                sm = SMOTE()
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X, y = sm.fit_resample(X, y)
                df = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=[target_col])], axis=1)
                st.success("✅ SMOTE applied")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        models = {
            "classification": {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVM": SVC(probability=True)
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR()
            }
        }

        results = []
        model_set = models[task_type]
        for name, model in model_set.items():
            model.fit(X, y)
            pred = model.predict(X)
            score = accuracy_score(y, pred) if task_type == "classification" else r2_score(y, pred)
            results.append((name, score))

        st.dataframe(pd.DataFrame(results, columns=["Model", "Score"]).sort_values("Score", ascending=False))

        best_model_name = max(results, key=lambda x: x[1])[0]
        best_model = model_set[best_model_name]
        st.success(f"🏆 Best Model: {best_model_name}")

        if role == "Yes" and st.radio("Want to tune manually?", ["No", "Yes"]) == "Yes":
            st.code("""
from sklearn.model_selection import GridSearchCV
params = {'max_depth': [3, 5, None]}
gscv = GridSearchCV(best_model, param_grid=params)
gscv.fit(X, y)
best_model = gscv.best_estimator_
            """, language="python")
        elif st.radio("Apply auto hyperparameter tuning?", ["No", "Yes"]) == "Yes":
            search_method = st.radio("Tuning method", ["GridSearchCV", "RandomizedSearchCV"])
            param_grid = {
                "Random Forest": {'n_estimators': [50, 100]},
                "Logistic Regression": {'C': [0.1, 1.0]},
                "SVM": {'C': [0.1, 1.0]},
                "Decision Tree": {'max_depth': [3, None]}
            }.get(best_model_name, {})

            search = GridSearchCV(best_model, param_grid=param_grid, cv=3) if search_method == "GridSearchCV" else RandomizedSearchCV(best_model, param_distributions=param_grid, n_iter=3, cv=3)
            search.fit(X, y)
            best_model = search.best_estimator_
            st.success("🎯 Hyperparameter tuning completed")

        final_pred = best_model.predict(X)
        if task_type == "classification":
            st.text(confusion_matrix(y, final_pred))
            st.text(classification_report(y, final_pred))
        else:
            st.write("R2 Score:", r2_score(y, final_pred))
            st.write("MAE:", mean_absolute_error(y, final_pred))

        with open("trained_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        st.success("✅ Model saved as trained_model.pkl")

        st.subheader("🧠 Predict and Summarize")
        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}")

        if st.button("Predict and Summarize"):
            input_df = pd.DataFrame([user_input])
            pred = best_model.predict(input_df)[0]
            decoded = label_encoders[target_col].inverse_transform([int(pred)])[0] if target_col in label_encoders else pred
            st.write(f"🔮 Predicted Output: {decoded}")
            summarizer = hf_pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
            summary_text = f"Using {best_model_name} with inputs {user_input}, prediction is {decoded}. Sweetviz revealed key feature patterns."
            summary = summarizer(summary_text, max_length=100, min_length=30, do_sample=False)
            st.write("📝 Summary:")
            st.write(summary[0]['summary_text'])
