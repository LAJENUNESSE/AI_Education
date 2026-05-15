"""可视化模块 - 结果可视化图表."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_accuracy_comparison(results, dataset_name, save=True):
    """绘制算法精度对比柱状图."""
    ensure_results_dir()

    names = []
    accuracies = []
    f1_scores = []

    for name, r in results.items():
        names.append(name)
        accuracies.append(r.get('accuracy', np.nan))
        f1_scores.append(r.get('f1_score', np.nan))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral', edgecolor='white')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title(f'Algorithm Performance Comparison - {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1 + bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f'accuracy_comparison_{dataset_name}.png')
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_training_time_comparison(results, dataset_name, save=True):
    """绘制训练时间对比图."""
    ensure_results_dir()

    names = [n for n, r in results.items()
             if not np.isnan(r.get('training_time', np.nan))]
    times = [r['training_time'] for n, r in results.items()
             if not np.isnan(r.get('training_time', np.nan))]

    if not times:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, times, color=colors, edgecolor='white')
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title(f'Training Time Comparison - {dataset_name}')

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times)*0.01, bar.get_y() + bar.get_height()/2.,
                f'{t:.4f}s', va='center', fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f'training_time_{dataset_name}.png')
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_confusion_matrix(cm, labels, dataset_name, model_name, save=True):
    """绘制混淆矩阵热力图."""
    if cm.size == 0:
        return None

    ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name} ({dataset_name})')

    plt.tight_layout()
    if save:
        safe_name = model_name.replace(' ', '_')
        path = os.path.join(RESULTS_DIR, f'confusion_{dataset_name}_{safe_name}.png')
        plt.savefig(path, dpi=150)
        plt.close()
    return fig


def plot_overall_summary(all_dataset_results, save=True):
    """绘制跨数据集综合对比图."""
    ensure_results_dir()

    datasets = list(all_dataset_results.keys())
    if not datasets:
        return

    first_dataset_results = all_dataset_results[datasets[0]]
    model_names = list(first_dataset_results.keys())

    n_datasets = len(datasets)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 3), max(6, n_models * 0.6)))

    data_matrix = np.zeros((n_models, n_datasets))
    for i, ds in enumerate(datasets):
        for j, model in enumerate(model_names):
            data_matrix[j, i] = all_dataset_results[ds].get(model, {}).get('accuracy', np.nan)

    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=datasets, yticklabels=model_names,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Algorithm')
    ax.set_title('Cross-Dataset Accuracy Comparison Heatmap')

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'overall_summary_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
