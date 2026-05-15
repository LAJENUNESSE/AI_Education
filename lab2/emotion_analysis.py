"""情感分析模块 - KNN/ANN/CNN 面部表情识别 (7 类情感)."""

import os
import pickle
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


EMOTION_LABEL_MAP = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}


def _load_from_csv(csv_path):
    """从 CSV 文件加载 FER2013."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    pixels = np.array([np.fromstring(p, dtype=np.float32, sep=' ')
                       for p in df['pixels']])
    X = pixels.reshape(-1, 48, 48, 1) / 255.0
    y = df['emotion'].values

    mask_train = df['Usage'] == 'Training'
    mask_test = df['Usage'] == 'PublicTest'
    mask_val = df['Usage'] == 'PrivateTest'

    return {
        'X_train': X[mask_train], 'y_train': y[mask_train],
        'X_val': X[mask_val], 'y_val': y[mask_val],
        'X_test': X[mask_test], 'y_test': y[mask_test]
    }


def _load_from_folder(base_dir):
    """从文件夹加载 FER2013 (Kaggle 图片格式).
    目录结构应为: base_dir/train/<emotion>/*.jpg 和 base_dir/test/<emotion>/*.jpg
    """
    import cv2

    def load_split(split):
        X_list, y_list = [], []
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            return [], []

        for emotion_name, label in EMOTION_LABEL_MAP.items():
            emo_dir = os.path.join(split_dir, emotion_name)
            if not os.path.isdir(emo_dir):
                continue
            for fname in os.listdir(emo_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(emo_dir, fname), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        X_list.append(img.reshape(48, 48, 1).astype(np.float32) / 255.0)
                        y_list.append(label)

        return np.array(X_list), np.array(y_list)

    X_train, y_train = load_split('train')
    X_test, y_test = load_split('test')

    if len(X_train) == 0:
        raise FileNotFoundError(f"未在 {base_dir}/train/ 下找到图片，请检查目录结构")

    # 从训练集划出 15% 做验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"  训练集: {X_train.shape[0]} 张, 验证集: {X_val.shape[0]} 张, 测试集: {X_test.shape[0]} 张")
    return {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test}


def load_fer2013():
    """加载 FER2013 数据集 (支持 CSV 和 Kaggle 文件夹两种格式)."""

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1) 先检查 CSV 文件
    csv_path = os.path.join(data_dir, 'fer2013.csv')
    if os.path.exists(csv_path):
        print(f"从 CSV 加载: {csv_path}")
        return _load_from_csv(csv_path)

    # 2) 再检查文件夹格式 (Kaggle 下载解压后)
    folder_path = os.path.join(data_dir, 'fer2013')
    if os.path.isdir(folder_path):
        print(f"从文件夹加载: {folder_path}")
        return _load_from_folder(folder_path)

    # 3) 尝试自动下载 CSV
    import tensorflow as tf
    urls = [
        'https://huggingface.co/datasets/tommyks/fer2013/resolve/main/fer2013.csv',
    ]
    for url in urls:
        try:
            print(f"尝试下载: {url}")
            tf.keras.utils.get_file('fer2013.csv', url, cache_dir=data_dir, cache_subdir='.')
            import glob
            for f in glob.glob(os.path.join(data_dir, 'fer2013*.csv')):
                if os.path.basename(f) == 'fer2013.csv' and f != csv_path:
                    import shutil
                    shutil.move(f, csv_path)
            if os.path.exists(csv_path):
                return _load_from_csv(csv_path)
        except Exception as e:
            print(f"  失败: {e}")

    raise FileNotFoundError(
        "FER2013 数据集未找到。请按以下任一种方式准备：\n\n"
        "方式A (推荐) - Kaggle 文件夹格式：\n"
        "  1. 浏览器打开 https://www.kaggle.com/datasets/msambare/fer2013\n"
        "  2. 点 Download 下载 archive.zip 并解压\n"
        "  3. 把解压后的 train/ 和 test/ 文件夹放到:\n"
        f"     {folder_path}/\n"
        "  4. 确保目录结构为: fer2013/train/angry/*.jpg 等\n\n"
        "方式B - CSV 格式：\n"
        f"  将 fer2013.csv 放到: {csv_path}\n"
    )


def train_knn_emotion(data=None):
    """训练 KNN 情感分类器."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report

    if data is None:
        data = load_fer2013()

    X_train = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)

    print("训练 KNN 情感分类器 ...")
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, data['y_train'])

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == data['y_test'])
    print(f"KNN 测试准确率: {acc:.4f}")
    print(classification_report(data['y_test'], y_pred, target_names=EMOTIONS))

    path = os.path.join(MODEL_DIR, 'emotion_knn.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return model, acc


def train_ann_emotion(data=None, epochs=30):
    """训练 ANN 情感分类器."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    if data is None:
        data = load_fer2013()

    X_train = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_val = data['X_val'].reshape(data['X_val'].shape[0], -1)
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    print("训练 ANN 情感分类器 ...")
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(2304,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=128, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"ANN 测试准确率: {acc:.4f}")

    path = os.path.join(MODEL_DIR, 'emotion_ann.h5')
    model.save(path)
    return model, acc, history


def train_cnn_emotion(data=None, epochs=50):
    """训练 CNN 情感分类器."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                          BatchNormalization, Flatten,
                                          RandomFlip, RandomRotation, RandomZoom)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    if data is None:
        data = load_fer2013()

    X_train = data['X_train'].reshape(-1, 48, 48, 1)
    X_val = data['X_val'].reshape(-1, 48, 48, 1)
    X_test = data['X_test'].reshape(-1, 48, 48, 1)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    print("训练 CNN 情感分类器 ...")

    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    model = Sequential([
        data_augmentation,
        Conv2D(64, (3, 3), activation='relu', padding='same',
               input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=64, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN 测试准确率: {acc:.4f}")

    path = os.path.join(MODEL_DIR, 'emotion_cnn.h5')
    model.save(path)
    return model, acc, history


def create_demo_emotion_model():
    """创建演示用情感分析模型 (随机权重 CNN，仅用于演示代码流程)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    path = os.path.join(MODEL_DIR, 'emotion_cnn_demo.h5')
    model.save(path)
    print("演示情感模型已创建 (CNN, 随机权重)")
    return model


def load_emotion_model(model_type='cnn'):
    """加载已训练的情感模型."""
    if model_type == 'knn':
        path = os.path.join(MODEL_DIR, 'emotion_knn.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    else:
        try:
            from tensorflow.keras.models import load_model
            path = os.path.join(MODEL_DIR, f'emotion_{model_type}.h5')
            if os.path.exists(path):
                return load_model(path)
        except Exception:
            pass
    return None


def train_all_emotion_models():
    """训练并比较所有情感分类模型."""
    data = load_fer2013()

    print("\n" + "="*50)
    print("  情感分析模型训练")
    print("="*50)

    results = {}

    try:
        _, acc_knn = train_knn_emotion(data)
        results['KNN'] = acc_knn
    except Exception as e:
        print(f"KNN 训练失败: {e}")

    try:
        _, acc_ann, _ = train_ann_emotion(data, epochs=30)
        results['ANN'] = acc_ann
    except Exception as e:
        print(f"ANN 训练失败: {e}")

    try:
        _, acc_cnn, _ = train_cnn_emotion(data, epochs=50)
        results['CNN'] = acc_cnn
    except Exception as e:
        print(f"CNN 训练失败: {e}")

    print("\n" + "-"*40)
    print("情感分析模型对比:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
    print("-"*40)

    return results


def predict_emotion(model, face_img, model_type='cnn'):
    """对单张人脸图像预测情感."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))

    if model_type == 'knn':
        x = resized.reshape(1, -1) / 255.0
        pred = model.predict(x)[0]
    else:
        x = resized.reshape(1, 48, 48, 1) / 255.0
        pred = np.argmax(model.predict(x, verbose=0)[0])

    return EMOTIONS[pred], pred
