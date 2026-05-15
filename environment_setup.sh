#!/bin/bash
# environment_setup.sh - 实验一环境配置脚本
set -e
cd "$(dirname "$0")/.."
if [ ! -d ".venv" ]; then
    uv venv --python 3.9 .venv
    source .venv/bin/activate
    export UV_HTTP_TIMEOUT=300
    uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        numpy scipy pandas matplotlib seaborn \
        jupyter notebook jupyterlab \
        scikit-learn xgboost lightgbm \
        tensorflow torch torchvision torchaudio \
        plotly ipykernel opencv-python dlib face_recognition
    python -m ipykernel install --user --name ai_edu --display-name "Python 3.9 (AI_Education)"
    echo "✅ 环境搭建完成"
else
    echo "⚠️ .venv 已存在，跳过创建"
fi
