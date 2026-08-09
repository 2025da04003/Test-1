import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
REPORT_DIR = BASE_DIR / 'reports'
VIZ_DIR = BASE_DIR / 'visualizations'
MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True); VIZ_DIR.mkdir(exist_ok=True)

def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy(); y = data.target.copy()
    full_df = X.copy(); full_df['target'] = y; full_df['target_name'] = full_df['target'].map({0:'malignant', 1:'benign'})
    cleaning_summary = []
    cleaning_summary.append(['Raw rows', full_df.shape[0]])
    cleaning_summary.append(['Raw columns', full_df.shape[1]])
    cleaning_summary.append(['Missing values before cleaning', int(full_df.isna().sum().sum())])
    cleaning_summary.append(['Duplicate rows before cleaning', int(full_df.duplicated().sum())])
    clean_df = full_df.drop_duplicates().copy()
    feature_cols = list(data.feature_names)
    for col in feature_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    cleaning_summary.append(['Rows after duplicate removal', clean_df.shape[0]])
    cleaning_summary.append(['Missing values after cleaning', int(clean_df.isna().sum().sum())])
    cleaning_summary.append(['Duplicate rows after cleaning', int(clean_df.duplicated().sum())])
    cleaning_summary.append(['Feature columns used for modelling', len(feature_cols)])
    pd.DataFrame(cleaning_summary, columns=['Check','Result']).to_csv(REPORT_DIR / 'data_cleaning_summary.csv', index=False)
    clean_df.to_csv(BASE_DIR / 'breast_cancer_full_dataset.csv', index=False)

    counts = clean_df['target_name'].value_counts().reindex(['malignant','benign'])
    plt.bar(counts.index, counts.values, color=['#fc8d62','#66c2a5'])
    plt.title('Class Distribution'); plt.xlabel('Diagnosis Class'); plt.ylabel('Number of Records')
    plt.tight_layout(); plt.savefig(VIZ_DIR / 'class_distribution.png', dpi=150); plt.close()
    corr = clean_df[feature_cols].corr()
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto'); plt.colorbar(label='Correlation'); plt.xticks([]); plt.yticks([])
    plt.title('Feature Correlation Heatmap'); plt.tight_layout(); plt.savefig(VIZ_DIR / 'correlation_heatmap.png', dpi=150); plt.close()
    groups = [clean_df.loc[clean_df['target_name']=='malignant','mean radius'], clean_df.loc[clean_df['target_name']=='benign','mean radius']]
    plt.boxplot(groups, labels=['malignant','benign'], patch_artist=True)
    plt.title('Mean Radius by Diagnosis Class'); plt.xlabel('Diagnosis Class'); plt.ylabel('Mean Radius')
    plt.tight_layout(); plt.savefig(VIZ_DIR / 'mean_radius_boxplot.png', dpi=150); plt.close()
    plt.figure(figsize=(8,5))
    class_colors = {'malignant':'#fc8d62', 'benign':'#66c2a5'}
    for label, group in clean_df.groupby('target_name'):
        plt.scatter(group['mean radius'], group['mean texture'], label=label, alpha=0.75, s=28, c=class_colors.get(label, '#999999'))
    plt.title('Scatter Plot: Mean Radius vs Mean Texture'); plt.xlabel('Mean Radius'); plt.ylabel('Mean Texture')
    plt.legend(title='Diagnosis Class'); plt.tight_layout(); plt.savefig(VIZ_DIR / 'scatter_mean_radius_texture.png', dpi=150); plt.close()
    plt.figure(figsize=(8,5))
    plt.hist(clean_df.loc[clean_df['target_name']=='malignant','mean radius'], bins=22, alpha=0.70, label='malignant', color='#fc8d62')
    plt.hist(clean_df.loc[clean_df['target_name']=='benign','mean radius'], bins=22, alpha=0.70, label='benign', color='#66c2a5')
    plt.title('Histogram: Mean Radius Distribution by Diagnosis Class'); plt.xlabel('Mean Radius'); plt.ylabel('Frequency')
    plt.legend(title='Diagnosis Class'); plt.tight_layout(); plt.savefig(VIZ_DIR / 'histogram_mean_radius.png', dpi=150); plt.close()

    X_clean = clean_df[feature_cols]; y_clean = clean_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clean)
    test_df = X_test.copy(); test_df['target'] = y_test.values; test_df.to_csv(BASE_DIR / 'test_data.csv', index=False)
    models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        'kNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced'),
    }
    files = {'Logistic Regression':'logistic_regression.joblib','Decision Tree':'decision_tree.joblib','kNN':'knn.joblib','Naive Bayes':'naive_bayes.joblib','Random Forest (Ensemble)':'random_forest_ensemble.joblib'}
    rows = []; all_reports = {}
    plt.figure(figsize=(7,6))
    for name, model in models.items():
        model.fit(X_train, y_train); joblib.dump(model, MODEL_DIR / files[name])
        pred = model.predict(X_test); prob = model.predict_proba(X_test)[:, 1]
        rows.append({'ML Model Name': name, 'Accuracy': accuracy_score(y_test,pred), 'AUC': roc_auc_score(y_test, prob), 'Precision': precision_score(y_test,pred,zero_division=0), 'Recall': recall_score(y_test,pred,zero_division=0), 'F1': f1_score(y_test,pred,zero_division=0), 'MCC': matthews_corrcoef(y_test,pred)})
        all_reports[name] = {'confusion_matrix': confusion_matrix(y_test,pred).tolist(), 'classification_report': classification_report(y_test,pred,output_dict=True,target_names=['malignant','benign'],zero_division=0)}
        fpr, tpr, _ = roc_curve(y_test, prob); plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr,tpr):.3f})')
    plt.plot([0,1],[0,1],'k--',label='Random Classifier'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves for Classification Models'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(VIZ_DIR / 'roc_curves.png', dpi=150); plt.close()
    metrics_df = pd.DataFrame(rows)
    for col in ['Accuracy','AUC','Precision','Recall','F1','MCC']: metrics_df[col] = metrics_df[col].round(4)
    metrics_df.to_csv(REPORT_DIR / 'model_metrics.csv', index=False)
    with open(REPORT_DIR / 'classification_reports.json', 'w') as f: json.dump(all_reports, f, indent=2)
    print(metrics_df.to_string(index=False))
if __name__ == '__main__': main()
