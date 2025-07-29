import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
import torch
import torch.nn.functional as F
import onnxruntime as ort
import os
import matplotlib.pyplot as plt
from cnn_model import CNN1D_Residual

st.set_page_config(page_title="Universal Model Deployment", layout="centered")
st.title("Universal Model Deployment")

model_file = st.file_uploader("Upload model file (.pkl / .pt / .onnx)", type=["pkl", "pt", "onnx"])
config_file = st.file_uploader("Upload configuration file (.json)", type=["json"])

model = None
predict_fn = None
model_type = None
config = None

if model_file and config_file:
    config = json.load(config_file)
    input_dim = config.get("input_dim", 2400)
    task = config.get("task", "classification")
    class_labels = config.get("class_labels", {"0": "Negative", "1": "Positive"})

    # Load model
    ext = os.path.splitext(model_file.name)[-1]
    if ext == ".pkl":
        model_type = "pkl"
        model = joblib.load(model_file)
        if hasattr(model, "predict_proba"):
            predict_fn = lambda X: model.predict_proba([X])[0]
        else:
            predict_fn = lambda X: model.predict([X])[0]

    elif ext == ".pt":
        model_type = "pt"
        device = torch.device("cpu")
        model = CNN1D_Residual(input_length=input_dim)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        def predict_fn(X):
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(X_tensor)
                prob = F.softmax(output, dim=1).squeeze().numpy()
            return prob

    elif ext == ".onnx":
        model_type = "onnx"
        session = ort.InferenceSession(model_file.read())

        def predict_fn(X):
            X = np.array(X, dtype=np.float32).reshape(1, 1, input_dim)
            inputs = {session.get_inputs()[0].name: X}
            outputs = session.run(None, inputs)
            return outputs[0].squeeze()

    st.success("Model and configuration file loaded successfully.")

    # Upload data
    st.subheader("Upload input data (CSV / XLSX)")
    data_file = st.file_uploader("Upload raw signal file", type=["csv", "xlsx"])

    if data_file:
        col_index = st.number_input("Select the column index containing signal values (0-based)", min_value=0, value=1, step=1)

        try:
            df_raw = pd.read_csv(data_file, skiprows=1) if data_file.name.endswith(".csv") else pd.read_excel(data_file, skiprows=1)
            st.write("Data preview (header skipped):", df_raw.head())

            if col_index >= df_raw.shape[1]:
                st.error("Selected column index is out of range.")
            elif st.button("Start Prediction"):
                raw_series = df_raw.iloc[:, col_index].dropna().astype(float).values
                sequence = raw_series[:input_dim]
                if len(sequence) < input_dim:
                    sequence = np.pad(sequence, (0, input_dim - len(sequence)))

                normalized = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

                # Plot normalized data only
                st.subheader("Standardized Input Signal")
                st.line_chart(normalized)

                # Predict
                probs = predict_fn(normalized)
                pred_class = int(np.argmax(probs))
                label = class_labels.get(str(pred_class), str(pred_class))

                st.success("Prediction completed.")
                st.markdown(f"**Predicted class: {label}**")
                st.markdown("Class probabilities:")
                for i, p in enumerate(probs):
                    st.markdown(f"- {class_labels.get(str(i), str(i))}: {p:.2%}")

                # Pie chart of probability
                st.subheader("Prediction probability pie chart")
                fig, ax = plt.subplots()
                ax.pie(probs, labels=[class_labels.get(str(i), str(i)) for i in range(len(probs))],
                       autopct='%1.1f%%', startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Data processing error: {e}")

elif model_file or config_file:
    st.warning("Please upload both model and configuration file.")
else:
    st.info("To start, please upload a model file and a configuration file.")
