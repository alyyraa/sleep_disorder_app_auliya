from models.train_model import train_models_pipeline
import pandas as pd

print('=== TRAINING MODEL WITH OCCUPATION INCLUDED ===')
trainer, clf_results, reg_results = train_models_pipeline('data/Sleep_health_and_lifestyle_dataset.csv')

if clf_results:
    print('\n=== CLASSIFICATION RESULTS ===')
    print(f'Accuracy: {clf_results["accuracy"]:.4f}')
    print(f'Precision: {clf_results["precision"]:.4f}')
    print(f'Recall: {clf_results["recall"]:.4f}')
    print(f'F1 Score: {clf_results["f1_score"]:.4f}')
    print('\nClassification Report:')
    print(clf_results['classification_report'])
    print('\nConfusion Matrix:')
    print(clf_results['confusion_matrix'])