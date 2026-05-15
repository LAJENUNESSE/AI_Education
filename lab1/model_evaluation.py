"""模型评估模块 - 统一评估框架."""

import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import warnings

warnings.filterwarnings('ignore')


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """统一评估机器学习模型性能."""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    try:
        accuracy = accuracy_score(y_test, y_pred)
    except Exception:
        accuracy = np.nan

    try:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    except Exception:
        precision = np.nan

    try:
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    except Exception:
        recall = np.nan

    try:
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    except Exception:
        f1 = np.nan

    try:
        cm = confusion_matrix(y_test, y_pred)
    except Exception:
        cm = np.array([])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


def compare_models(model_dict, X_train, X_test, y_train, y_test):
    """比较多个模型在相同数据集上的性能."""
    results = {}

    for name, model in model_dict.items():
        print(f"    评估: {name} ...", end=' ')
        try:
            res = evaluate_model(model, X_train, X_test, y_train, y_test)
            results[name] = res
            print(f"✓ (acc={res['accuracy']:.4f})")
        except Exception as e:
            print(f"✗ ({e})")
            results[name] = {
                'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan,
                'f1_score': np.nan, 'training_time': np.nan,
                'prediction_time': np.nan, 'confusion_matrix': np.array([]),
                'y_pred': None
            }

    return results


def build_model_dict(input_dim, n_classes):
    """构建所有待评估模型的字典."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  VotingClassifier, GradientBoostingClassifier)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    return models


def print_results_table(results):
    """打印结果汇总表."""
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train(s)', 'Pred(s)']
    print(f"\n{'='*85}")
    print(f"{'Model':<25s} {'Accuracy':>8s} {'Precision':>8s} {'Recall':>8s} {'F1':>8s} {'Train':>8s} {'Pred':>8s}")
    print('-'*85)

    for name, r in results.items():
        print(f"{name:<25s} "
              f"{r['accuracy']:8.4f} "
              f"{r['precision']:8.4f} "
              f"{r['recall']:8.4f} "
              f"{r['f1_score']:8.4f} "
              f"{r['training_time']:8.4f} "
              f"{r['prediction_time']:8.4f}")
    print('-'*85)
