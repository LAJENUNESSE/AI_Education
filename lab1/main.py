"""实验一主程序 - 机器学习开发平台搭建与算法测试."""

import os
import sys
import time
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CUDA Toolkit 已安装，确保 libdevice.10.bc 可被 XLA 找到
_libdevice_src = '/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc'
_libdevice_dst = os.path.join(os.path.dirname(__file__), 'libdevice.10.bc')
if os.path.exists(_libdevice_src) and not os.path.exists(_libdevice_dst):
    os.symlink(_libdevice_src, _libdevice_dst)
# 清除 apt 代理，避免干扰 Python HTTPS 请求
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import get_all_datasets, split_and_prepare
from model_evaluation import (build_model_dict, compare_models,
                               print_results_table)
from visualization import (plot_accuracy_comparison,
                            plot_training_time_comparison,
                            plot_confusion_matrix,
                            plot_overall_summary)

from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)


def create_dnn_model(input_dim, output_dim):
    """创建深度神经网络模型."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_dnn(model, X_train, X_test, y_train, y_test,
                 epochs=50, batch_size=32, verbose=0):
    """评估 DNN 模型."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ]
    start_time = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=verbose, validation_split=0.1, callbacks=callbacks)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    prediction_time = time.time() - start_time

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'training_time': training_time,
        'prediction_time': prediction_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }


def run_experiment(datasets):
    """运行完整实验流程."""
    print("=" * 60)
    print("  机器学习算法对比实验")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_dataset_results = {}

    for ds_name, (X, y, target_names) in datasets.items():
        print(f"\n{'='*60}")
        print(f"  数据集: {ds_name}")
        print(f"  样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")
        print(f"{'='*60}")

        n_classes = len(np.unique(y))
        if n_classes > 15:
            print(f"  [跳过] 类别过多 ({n_classes}), 不适合当前分类器")
            continue

        use_pca = X.shape[1] > 2000
        X_train, X_test, y_train, y_test = split_and_prepare(X, y,
            test_size=0.2, apply_pca=use_pca, n_components=200)
        print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 传统算法
        models = build_model_dict(X_train.shape[1], n_classes)
        results = compare_models(models, X_train, X_test, y_train, y_test)

        # DNN
        print("    评估: DNN ...", end=' ')
        try:
            dnn = create_dnn_model(X_train.shape[1], n_classes)
            dnn_res = evaluate_dnn(dnn, X_train, X_test, y_train, y_test)
            results['DNN'] = dnn_res
            print(f"✓ (acc={dnn_res['accuracy']:.4f})")
        except Exception as e:
            print(f"✗ ({e})")

        all_dataset_results[ds_name] = results
        print_results_table(results)

        # 可视化
        try:
            plot_accuracy_comparison(results, ds_name)
            plot_training_time_comparison(results, ds_name)

            sorted_models = sorted(results.items(),
                                   key=lambda x: x[1].get('accuracy', 0), reverse=True)
            for model_name, r in sorted_models[:3]:
                cm = r.get('confusion_matrix', np.array([]))
                if cm.size > 0:
                    labels_subset = target_names[:cm.shape[0]]
                    plot_confusion_matrix(cm, labels_subset, ds_name, model_name)
        except Exception as e:
            print(f"  [可视化警告] {e}")

    # 跨数据集综合对比
    if len(all_dataset_results) > 1:
        try:
            plot_overall_summary(all_dataset_results)
        except Exception as e:
            print(f"  [综合可视化警告] {e}")

    print(f"\n{'='*60}")
    print(f"  实验完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结果保存在: {os.path.dirname(__file__)}/results/")
    print(f"{'='*60}")

    return all_dataset_results


def main():
    """主函数."""
    datasets = get_all_datasets()
    print(f"\n已加载 {len(datasets)} 个数据集:")
    for name, (X, y, _) in datasets.items():
        print(f"  - {name}: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类")

    results = run_experiment(datasets)

    print("\n实验总结:")
    print("-" * 60)
    for ds_name, ds_results in results.items():
        if ds_results:
            best_model = max(ds_results.items(),
                             key=lambda x: x[1].get('accuracy', 0))
            print(f"  {ds_name}: 最佳算法 {best_model[0]} "
                  f"(acc={best_model[1]['accuracy']:.4f})")


if __name__ == '__main__':
    main()
