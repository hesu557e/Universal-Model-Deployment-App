import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# 初始化目录
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 页面标题
st.set_page_config(page_title="通用模型部署平台", layout="centered")
st.title("🧠 通用机器学习模型部署平台")

# 上传模型
model_file = st.file_uploader("📁 上传模型文件 (.pkl)", type=["pkl"])
config_file = st.file_uploader("⚙️ 上传模型配置 (.json)", type=["json"])

# 状态变量
model = None
config = None

if model_file and config_file:
    model_path = os.path.join("models", model_file.name)
    with open(model_path, "wb") as f:
        f.write(model_file.read())
    model = joblib.load(model_path)

    config = json.load(config_file)
    input_dim = config.get("input_dim")
    input_labels = config.get("input_labels", [f"Feature {i+1}" for i in range(input_dim)])
    task = config.get("task", "classification")
    class_labels = config.get("class_labels", {})
    threshold = config.get("threshold", 0.5)

    st.success("✅ 模型和配置加载成功")

    # 输入区域
    st.subheader("✍️ 输入特征值")
    inputs = []
    cols = st.columns(2)
    for i in range(input_dim):
        val = cols[i % 2].number_input(f"{input_labels[i]}", value=0.0)
        inputs.append(val)

    if st.button("🚀 开始预测"):
        X = np.array(inputs).reshape(1, -1)
        if hasattr(model, "predict_proba") and task == "classification":
            proba = model.predict_proba(X)[0]
            pred_class = np.argmax(proba)
            label = class_labels.get(str(pred_class), str(pred_class))
            st.success(f"预测类别：**{label}**")
            st.write("预测概率：")
            st.bar_chart(pd.Series(proba, index=[class_labels.get(str(i), str(i)) for i in range(len(proba))]))
        else:
            result = model.predict(X)[0]
            st.success(f"预测值：**{result}**")

        # 日志记录
        log = pd.DataFrame([{
            "time": datetime.now().isoformat(),
            **{input_labels[i]: inputs[i] for i in range(input_dim)},
            "prediction": label if task == "classification" else result
        }])
        log_path = os.path.join("logs", "predict_log.csv")
        if os.path.exists(log_path):
            pd.read_csv(log_path).append(log).to_csv(log_path, index=False)
        else:
            log.to_csv(log_path, index=False)

elif model_file or config_file:
    st.warning("⚠️ 请同时上传模型文件和配置文件")

