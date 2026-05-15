"""算法实现模块 - 实现和测试多种机器学习算法."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              VotingClassifier, GradientBoostingClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import warnings

warnings.filterwarnings('ignore')


def test_knn(X, y):
    """K近邻算法测试."""
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    return scores.mean()


def test_linear_models(X, y, problem_type='classification'):
    """线性回归与逻辑回归测试."""
    if problem_type == 'classification':
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        metric_name = 'accuracy'
        mean_score = scores.mean()
    else:
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        metric_name = 'r2'
        mean_score = scores.mean()

    return mean_score


def test_tree_bayes(X, y):
    """决策树与朴素贝叶斯分类器测试."""
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    nb = GaussianNB()

    dt_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    nb_scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy')

    return {
        'decision_tree': dt_scores.mean(),
        'naive_bayes': nb_scores.mean()
    }


def test_svm_clustering(X, y):
    """支持向量机与K-Means聚类测试."""
    n_classes = len(np.unique(y))
    n_clusters = max(2, n_classes)

    svm = SVC(kernel='rbf', random_state=42)
    svm_scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, km_labels)

    return {
        'svm': svm_scores.mean(),
        'kmeans_silhouette': sil_score
    }


def test_ensemble_methods(X, y):
    """集成学习方法测试."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('ada', ada), ('gb', gb)],
        voting='soft'
    )

    results = {}
    for name, model in [('RandomForest', rf), ('AdaBoost', ada),
                         ('GradientBoosting', gb), ('VotingClassifier', voting_clf)]:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = scores.mean()

    return results
