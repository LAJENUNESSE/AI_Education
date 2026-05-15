"""数据预处理模块 - 加载和预处理实验所需数据集."""

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


def load_sklearn_datasets():
    """加载 scikit-learn 内置数据集."""
    datasets = {}

    iris = load_iris()
    datasets['iris'] = (iris.data, iris.target, iris.target_names)

    digits = load_digits()
    datasets['digits'] = (digits.data, digits.target, digits.target_names.astype(str))

    cancer = load_breast_cancer()
    datasets['breast_cancer'] = (cancer.data, cancer.target, cancer.target_names.astype(str))

    return datasets


def load_titanic():
    """加载 Titanic 数据集 (通过 seaborn 内置数据集)."""
    import pandas as pd
    import seaborn as sns

    df = sns.load_dataset('titanic')

    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)

    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    df['embarked'] = LabelEncoder().fit_transform(df['embarked'].astype(str))
    df['class'] = LabelEncoder().fit_transform(df['class'].astype(str))
    df['who'] = LabelEncoder().fit_transform(df['who'].astype(str))
    df['alone'] = df['alone'].astype(int)

    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
                    'class', 'who', 'alone']
    X = df[feature_cols].values.astype(np.float64)
    y = df['survived'].values
    target_names = np.array(['Not Survived', 'Survived'])

    return X, y, target_names


def load_mnist(samples=5000):
    """加载 MNIST 数据集 (子集以控制运行时间)."""
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])[:samples]
    y = np.concatenate([y_train, y_test])[:samples]
    X = X.reshape(X.shape[0], -1).astype(np.float64) / 255.0
    target_names = np.array([str(i) for i in range(10)])
    return X, y, target_names


def load_cifar10(samples=5000):
    """加载 CIFAR-10 数据集 (子集以控制运行时间)."""
    from tensorflow.keras.datasets import cifar10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X = np.concatenate([X_train, X_test])[:samples]
    y = np.concatenate([y_train, y_test])[:samples]
    X = X.reshape(X.shape[0], -1).astype(np.float64) / 255.0
    y = y.ravel()
    target_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'])
    return X, y, target_names


def load_external_datasets():
    """加载外部数据集."""
    datasets = {}

    datasets['titanic'] = load_titanic()

    try:
        datasets['mnist'] = load_mnist(samples=5000)
    except Exception as e:
        print(f"  [警告] MNIST 加载失败: {e}")

    try:
        datasets['cifar10'] = load_cifar10(samples=5000)
    except Exception as e:
        print(f"  [警告] CIFAR-10 加载失败: {e}")

    return datasets


def preprocess_data(X, y, apply_pca=False, n_components=50):
    """标准化数据，可选 PCA 降维."""
    y = y.astype(np.int64)

    if apply_pca and X.shape[1] > n_components:
        n_components = min(n_components, X.shape[0], X.shape[1])
        X = PCA(n_components=n_components, random_state=42).fit_transform(X)

    return X, y


def split_and_prepare(X, y, test_size=0.2, apply_pca=False, n_components=50):
    """划分训练/测试集并标准化."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train, y_train = preprocess_data(X_train, y_train, apply_pca, n_components)
    X_test, y_test = preprocess_data(X_test, y_test, apply_pca, n_components)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def get_all_datasets():
    """获取所有可用数据集."""
    datasets = load_sklearn_datasets()

    print("加载外部数据集...")
    external = load_external_datasets()
    datasets.update(external)

    return datasets
