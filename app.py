import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title='Breast Cancer Classification Assignment', layout='wide')
st.title('Machine Learning Assignment 2 - Breast Cancer Classification')
st.caption('Target encoding: 0 = malignant, 1 = benign. This app is for academic ML demonstration only.')
st.success('Streamlit-ready package: upload test_data.csv in the Model Evaluation section after deployment.')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'kNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest (Ensemble)': 'random_forest_ensemble.joblib'
}

@st.cache_resource
def load_model(model_file):
    return joblib.load(Path('model') / model_file)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    }
    return metrics, y_pred

st.sidebar.header('Project Sections')
section = st.sidebar.radio('Select section', ['Data Cleaning', 'Visualisation', 'Model Evaluation'])

if section == 'Data Cleaning':
    st.header('Data Cleaning Summary')
    st.write('This section shows the data cleaning checks performed before model training.')
    summary_path = Path('reports/data_cleaning_summary.csv')
    if summary_path.exists():
        st.dataframe(pd.read_csv(summary_path), use_container_width=True)
    else:
        st.warning('data_cleaning_summary.csv was not found.')
    st.markdown('''
**Cleaning steps used**
- Checked missing values.
- Checked duplicate rows.
- Removed duplicates if found.
- Added median-imputation safety step for numerical features.
- Saved cleaned dataset as `breast_cancer_full_dataset.csv`.
- Created stratified test data as `test_data.csv`.
''')

elif section == 'Visualisation':
    st.header('Exploratory Data Visualisation')
    visual_items = [
        ('Class Distribution', 'visualizations/class_distribution.png'),
        ('Feature Correlation Heatmap', 'visualizations/correlation_heatmap.png'),
        ('Mean Radius Boxplot', 'visualizations/mean_radius_boxplot.png'),
        ('Histogram: Mean Radius Distribution', 'visualizations/histogram_mean_radius.png'),
        ('Scatter Plot: Mean Radius vs Mean Texture', 'visualizations/scatter_mean_radius_texture.png'),
        ('ROC Curve Comparison', 'visualizations/roc_curves.png')
    ]
    for title, image_path in visual_items:
        st.subheader(title)
        if Path(image_path).exists():
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f'Missing image: {image_path}')

else:
    st.header('Model Evaluation on Uploaded Test Data')
    st.markdown('''
This section satisfies the required Streamlit app features:
1. CSV dataset upload option
2. Model selection dropdown
3. Evaluation metrics display
4. Confusion matrix and classification report
5. Comparison results for all models on the uploaded test data
''')

    uploaded_file = st.file_uploader('Upload test CSV file. Use only test_data.csv for Streamlit free tier.', type=['csv'])
    selected_model = st.selectbox('Select a classification model', list(MODEL_FILES.keys()))

    if uploaded_file is None:
        st.info('Please upload test_data.csv from this repository to start evaluation.')
    else:
        test_data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Test Data Preview')
        st.dataframe(test_data.head(10), use_container_width=True)

        if 'target' not in test_data.columns:
            st.error('CSV must include a target column.')
            st.stop()

        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']

        st.subheader('Comparison Results for All Models on Uploaded Test Data')
        comparison_rows = []
        for model_name, model_file in MODEL_FILES.items():
            model = load_model(model_file)
            metrics, _ = evaluate_model(model, X_test, y_test)
            comparison_rows.append({
                'ML Model Name': model_name,
                'Accuracy': metrics['Accuracy'],
                'AUC': metrics['AUC Score'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1 Score'],
                'MCC': metrics['MCC Score']
            })
        comparison_df = pd.DataFrame(comparison_rows).round(4)
        st.dataframe(comparison_df, use_container_width=True)

        model = load_model(MODEL_FILES[selected_model])
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        st.subheader(f'Selected Model Evaluation Metrics - {selected_model}')
        cols = st.columns(3)
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            cols[idx % 3].metric(metric_name, f'{metric_value:.4f}')

        st.subheader(f'Confusion Matrix - {selected_model}')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['malignant', 'benign'], yticklabels=['malignant', 'benign'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)

        st.subheader(f'Classification Report - {selected_model}')
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['malignant', 'benign'], zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        prediction_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
        st.download_button('Download selected model predictions CSV', prediction_df.to_csv(index=False), 'predictions.csv', 'text/csv')
        st.download_button('Download all-model comparison CSV', comparison_df.to_csv(index=False), 'model_comparison_on_uploaded_test_data.csv', 'text/csv')
