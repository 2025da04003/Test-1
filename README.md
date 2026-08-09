# Machine Learning Assignment 2 - Breast Cancer Wisconsin Classification

## a. Problem statement
Implement, evaluate, visualize, and deploy multiple classification models for the Breast Cancer Wisconsin Diagnostic dataset. The Streamlit app includes data cleaning summary, visualisation, model selection, evaluation metrics, confusion matrix, classification report, and prediction download.

## b. Dataset description
- Dataset source: UCI Machine Learning Repository, accessed through Scikit-learn `load_breast_cancer()`
- Problem type: Binary classification
- Instances: 569
- Features: 30 real-valued features
- Target: `target`, where 0 = malignant and 1 = benign
- Requirement status: satisfies minimum 500 instances and 12 features

## c. Github Repository Link
Replace after upload: `https://github.com/<your-username>/<your-repository-name>`

## Live Streamlit App Link
Replace after deployment: `https://<your-app-name>.streamlit.app/`

## Data cleaning performed
1. Loaded full dataset and saved `breast_cancer_full_dataset.csv`.
2. Checked shape, missing values, and duplicate rows.
3. Removed duplicate rows if present.
4. Added a median-imputation safety step for numerical features.
5. Added readable target names for EDA: malignant and benign.
6. Used stratified train-test split and saved `test_data.csv`.

## Visualisation performed
- Class distribution chart
- Feature correlation heatmap
- Mean radius boxplot by diagnosis class
- ROC curve comparison for all models
- Scatter plot: Mean Radius vs Mean Texture
- Histogram: Mean Radius distribution by diagnosis class

## d. Models used and comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9371 | 0.9186 | 0.9551 | 0.9444 | 0.9497 | 0.8657 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9893 | 0.9263 | 0.9778 | 0.9514 | 0.8650 |
| Random Forest (Ensemble) | 0.9580 | 0.9949 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

## Model performance observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model, with the highest accuracy, AUC, F1, and MCC. |
| Decision Tree | Interpretable but less stable than ensemble and scaled linear models. |
| kNN | Very strong recall and F1 after feature scaling. |
| Naive Bayes | Fast baseline with good AUC but lower accuracy than Logistic Regression. |
| Random Forest (Ensemble) | Robust ensemble with strong AUC and balanced performance. |
| Overall Winner for this dataset | Logistic Regression. |

## Repository structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- breast_cancer_full_dataset.csv
|-- model/
|   |-- train_models.py
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|-- reports/
|   |-- data_cleaning_summary.csv
|   |-- model_metrics.csv
|   |-- classification_reports.json
|   |-- roc_curve_data.json
|   |-- Assignment_Report.pdf
|-- visualizations/
|   |-- class_distribution.png
|   |-- correlation_heatmap.png
|   |-- mean_radius_boxplot.png
|   |-- roc_curves.png
|   |-- scatter_mean_radius_texture.png
|   |-- histogram_mean_radius.png
```

## Step-by-step execution
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Final submission checklist
- GitHub repository link works
- Streamlit app link opens correctly
- App shows Data Cleaning, Visualisation, and Model Evaluation sections
- CSV upload works
- Metrics, confusion matrix, and classification report are visible
- README content is added to final PDF
- BITS Virtual Lab execution screenshot is inserted into final PDF

## Academic note
This project is for coursework only and should not be used as a medical diagnostic tool.


## Streamlit-ready deployment notes

This package is ready for Streamlit Community Cloud deployment.

1. Upload the full repository to GitHub.
2. Confirm these files are present in the root folder: `app.py`, `requirements.txt`, `test_data.csv`, `README.md`, and the `model/`, `reports/`, and `visualizations/` folders.
3. In Streamlit Community Cloud, select `app.py` as the main file.
4. After deployment, open the app, go to **Model Evaluation**, and upload `test_data.csv`.
5. Verify that **Data Cleaning**, **Visualisation**, and **Model Evaluation** sections load correctly.


## Streamlit app requirement mapping

| Assignment requirement | Implementation in this app |
|---|---|
| Dataset upload option, CSV | `st.file_uploader()` is available in the Model Evaluation section. Upload only `test_data.csv`. |
| Model selection dropdown | `st.selectbox()` lists Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest. |
| Display evaluation metrics | Accuracy, AUC, Precision, Recall, F1 Score, and MCC Score are shown as metric cards. |
| Confusion matrix or classification report | Both confusion matrix and classification report are displayed for the selected model. |
| Results of different models on test data | A comparison table evaluates all five models on the uploaded test data. |
