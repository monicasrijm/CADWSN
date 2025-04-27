import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app1.ml_models import models, load_models

# Load all models once
load_models()

# 🔮 GenAI: Generate explanations (mocked logic for now)
def generate_explanation(predictions, df):
    attack_count = np.sum(predictions)
    if attack_count == 0:
        return "✅ No anomalies detected. All features fall within expected thresholds."
    else:
        top_features = df.mean().sort_values(ascending=False).head(3).index.tolist()
        return (
            f"⚠️ Detected {attack_count} attack(s). "
            f"Influential metrics: {', '.join(top_features)}. "
            "Suggested action: Check for high traffic, abnormal energy consumption, or rerouting behavior."
        )

# 📌 Model recommendation based on data
def recommend_model(df):
    feature_std = df.std().mean()
    if feature_std < 1:
        return "SVM is suitable for this relatively consistent dataset."
    elif feature_std < 3:
        return "RandomForest may perform well due to moderate variance."
    else:
        return "NeuralNet is recommended for high-dimensional or noisy data."

# App layout
st.set_page_config(page_title="WSN Cyber Attack Detection (GenAI Enhanced)", layout="wide")
st.title("🔐 WSN Cyber Attack Detection with GenAI Intelligence")

st.sidebar.header("📁 Upload WSN Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    st.sidebar.header("⚙️ Select Detection Model")
    model_choice = st.sidebar.selectbox("Choose a model to predict with:", list(models.keys()))

    if st.button("🔍 Run Detection"):
        with st.spinner("🔄 Running predictions..."):
            model = models[model_choice]

            try:
                # 🧹 Preprocess
                if 'label' in df.columns:
                    df = df.drop(columns=['label'])

                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype('category').cat.codes

                df = df.astype(np.float32)

                expected_features = 10  # Customize as per your model input
                if df.shape[1] > expected_features:
                    df = df.iloc[:, :expected_features]
                elif df.shape[1] < expected_features:
                    raise ValueError(f"Expected {expected_features} features, got {df.shape[1]}.")

                input_data = torch.tensor(df.values, dtype=torch.float32)

                # Predict
                with torch.no_grad():
                    outputs = model(input_data).numpy()

                if outputs.shape[1] == 1:
                    predictions = (outputs > 0.5).astype(int).flatten()
                else:
                    predictions = np.argmax(outputs, axis=1)

                df["Prediction"] = predictions

                attack_count = np.sum(predictions)
                if attack_count > 0:
                    st.error(f"🚨 {attack_count} potential attacks detected!")
                else:
                    st.success("✅ No attacks detected.")

                # Explanation from GenAI logic
                st.subheader("🧠 GenAI Explanation")
                explanation = generate_explanation(predictions, df.drop(columns=['Prediction']))
                st.info(explanation)

                # Model Recommendation
                st.subheader("📈 Smart Model Recommendation")
                recommendation = recommend_model(df.drop(columns=['Prediction']))
                st.success(recommendation)

                # Display prediction
                st.subheader("📊 Prediction Results")
                st.dataframe(df)

                # Visualization
                st.subheader(f"📉 {model_choice} - Detection Visualization")
                if model_choice == "SVM":
                    sns.histplot(df["Prediction"], bins=2, kde=False)
                    plt.title("SVM Prediction Histogram")
                    st.pyplot(plt.gcf())
                    plt.clf()

                elif model_choice == "RandomForest":
                    st.markdown("#### Feature Importance (Simulated)")
                    fake_importance = np.random.rand(df.shape[1] - 1)
                    sns.barplot(x=fake_importance, y=df.columns[:-1])
                    plt.title("RandomForest Feature Importance")
                    st.pyplot(plt.gcf())
                    plt.clf()

                elif model_choice == "NeuralNet":
                    plt.plot(outputs[:100])
                    plt.title("NeuralNet Output Probabilities")
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    sns.countplot(x="Prediction", data=df)
                    plt.title("Prediction Count")
                    st.pyplot(plt.gcf())
                    plt.clf()

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

else:
    st.info("Please upload a CSV file to get started.")


# 🧠 Chatbot Assistant (Sidebar)
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **CyberSec Assistant**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.sidebar.text_input("Ask me anything...")

def get_response(prompt):
    prompt = prompt.lower()
    if "prediction" in prompt:
        return "Prediction '1' usually indicates an anomaly or attack, while '0' means normal behavior."
    elif "svm" in prompt:
        return "SVM (Support Vector Machine) is great for binary classification, especially in smaller or clean datasets."
    elif "prevent" in prompt or "mitigate" in prompt:
        return "To mitigate attacks: monitor node traffic, apply encryption, isolate suspicious nodes, and use intrusion detection systems."
    elif "best model" in prompt:
        return "Model performance depends on data variance. For low variance: SVM, medium: RandomForest, high: NeuralNet."
    else:
        return "I'm still learning! Try asking about predictions, models, or prevention."

if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("🧑", user_input))
    st.session_state.chat_history.append(("🤖", response))

    for speaker, msg in st.session_state.chat_history[-6:]:  # limit to last 3 pairs
        st.sidebar.markdown(f"**{speaker}**: {msg}")


# # TRIAL 2
# import streamlit as st
# import pandas as pd
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from app1.ml_models import models, load_models

# # Load models
# load_models()

# # 🔮 Attack labels - change according to your dataset
# attack_labels = {
#     0: "Normal",
#     1: "Sybil Attack",
#     2: "Blackhole Attack",
#     3: "Flooding Attack",
#     4: "Wormhole Attack"
# }

# # App layout
# st.set_page_config(page_title="WSN Attack Classification", layout="wide")
# st.title("🔐 WSN Cyber Attack Classification with GenAI & Visual Insights")

# st.sidebar.header("📁 Upload WSN Data")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📄 Uploaded Data Preview")
#     st.dataframe(df.head())

#     st.sidebar.header("⚙️ Select Detection Model")
#     model_choice = st.sidebar.selectbox("Choose a model:", list(models.keys()))

#     if st.button("🔍 Classify Attacks"):
#         with st.spinner("🔄 Classifying..."):
#             model = models[model_choice]

#             try:
#                 if 'label' in df.columns:
#                     df = df.drop(columns=['label'])

#                 for col in df.select_dtypes(include=['object']).columns:
#                     df[col] = df[col].astype('category').cat.codes

#                 df = df.astype(np.float32)

#                 expected_features = 10  # adjust as needed
#                 if df.shape[1] > expected_features:
#                     df = df.iloc[:, :expected_features]
#                 elif df.shape[1] < expected_features:
#                     raise ValueError(f"Expected {expected_features} features, got {df.shape[1]}.")

#                 input_data = torch.tensor(df.values, dtype=torch.float32)

#                 with torch.no_grad():
#                     outputs = model(input_data).numpy()

#                 # Predict attack type
#                 predictions = np.argmax(outputs, axis=1)
#                 df["Attack_Type"] = [attack_labels[pred] for pred in predictions]

#                 # Alert
#                 attack_detected = df["Attack_Type"] != "Normal"
#                 count_attacks = attack_detected.sum()
#                 if count_attacks > 0:
#                     st.error(f"🚨 {count_attacks} attacks detected in data!")
#                 else:
#                     st.success("✅ All data points appear normal.")

#                 # Display classified results
#                 st.subheader("📊 Classified Results")
#                 st.dataframe(df)

#                 # Save to CSV
#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button("📥 Download Classified Data", csv, "classified_attacks.csv", "text/csv")

#                 # 📈 Visualizations
#                 st.subheader("📉 Attack Type Visualizations")

#                 col1, col2 = st.columns(2)

#                 with col1:
#                     st.markdown("#### Bar Chart - Attack Types")
#                     plt.figure(figsize=(8, 4))
#                     sns.countplot(x="Attack_Type", data=df, palette="Set2")
#                     plt.xticks(rotation=45)
#                     st.pyplot(plt.gcf())
#                     plt.clf()

#                 with col2:
#                     st.markdown("#### Pie Chart - Attack Distribution")
#                     pie_data = df["Attack_Type"].value_counts()
#                     fig, ax = plt.subplots()
#                     ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
#                     ax.axis('equal')
#                     st.pyplot(fig)

#                 # Heatmap (optional)
#                 st.subheader("📌 Feature Correlation Heatmap")
#                 corr = df.drop(columns=['Attack_Type']).corr()
#                 plt.figure(figsize=(10, 6))
#                 sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
#                 st.pyplot(plt.gcf())
#                 plt.clf()

#             except Exception as e:
#                 st.error(f"❌ Error during classification: {e}")

# else:
#     st.info("Please upload a CSV file to get started.")


# import streamlit as st
# import pandas as pd
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from app1.ml_models import models, load_models

# # Load models
# load_models()

# # 🔖 Attack labels - update according to your model's outputs
# attack_labels = {
#     0: "Normal",
#     1: "Sybil Attack",
#     2: "Blackhole Attack",
#     3: "Flooding Attack",
#     4: "Wormhole Attack"
# }

# # 🔮 GenAI Explanation (mocked for demo)
# def generate_explanation(predictions, df):
#     attack_count = (predictions != 0).sum()
#     if attack_count == 0:
#         return "✅ No anomalies detected. All data points labeled as Normal."
#     else:
#         top_features = df.var().sort_values(ascending=False).head(3).index.tolist()
#         return (
#             f"⚠️ {attack_count} attack(s) detected. "
#             f"Common influential metrics: {', '.join(top_features)}. "
#             "Suggest reviewing nodes for abnormal patterns or protocol violations."
#         )

# # 🔍 Model recommendation
# def recommend_model(df):
#     feature_std = df.std().mean()
#     if feature_std < 1:
#         return "SVM is best for consistent datasets."
#     elif feature_std < 3:
#         return "RandomForest fits moderate variation."
#     else:
#         return "NeuralNet is ideal for complex or high-noise datasets."

# # Layout
# st.set_page_config(page_title="WSN Attack Type Classification", layout="wide")
# st.title("🔐 WSN Attack Type Classification with GenAI Insights")

# st.sidebar.header("📁 Upload WSN Data")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📄 Uploaded Data Preview")
#     st.dataframe(df.head())

#     st.sidebar.header("⚙️ Select Detection Model")
#     model_choice = st.sidebar.selectbox("Choose a model:", list(models.keys()))

#     if st.button("🔍 Classify Attacks"):
#         with st.spinner("🔄 Running classification..."):
#             try:
#                 model = models[model_choice]

#                 if 'label' in df.columns:
#                     df = df.drop(columns=['label'])

#                 # Encode categorical columns
#                 for col in df.select_dtypes(include='object').columns:
#                     df[col] = df[col].astype('category').cat.codes

#                 df = df.astype(np.float32)

#                 expected_features = 10
#                 if df.shape[1] > expected_features:
#                     df = df.iloc[:, :expected_features]
#                 elif df.shape[1] < expected_features:
#                     raise ValueError(f"Expected {expected_features} features, got {df.shape[1]}.")

#                 input_tensor = torch.tensor(df.values, dtype=torch.float32)

#                 with torch.no_grad():
#                     output = model(input_tensor).numpy()

#                 predictions = np.argmax(output, axis=1)
#                 df["Attack_Type"] = [attack_labels.get(pred, "Unknown") for pred in predictions]

#                 # Show results
#                 attack_detected = df["Attack_Type"] != "Normal"
#                 attack_count = attack_detected.sum()
#                 if attack_count > 0:
#                     st.error(f"🚨 {attack_count} attacks detected!")
#                 else:
#                     st.success("✅ All data points appear normal.")

#                 # Explanation
#                 st.subheader("🧠 GenAI Explanation")
#                 explanation = generate_explanation(predictions, df.drop(columns=["Attack_Type"]))
#                 st.info(explanation)

#                 # Recommendation
#                 st.subheader("📈 Smart Model Recommendation")
#                 st.success(recommend_model(df.drop(columns=["Attack_Type"])))

#                 # Display results
#                 st.subheader("📊 Classified Results")
#                 st.dataframe(df)

#                 # Download
#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button("📥 Download Results", csv, "classified_attacks.csv", "text/csv")

#                 # Visualizations
#                 st.subheader("📉 Attack Type Visualizations")
#                 col1, col2 = st.columns(2)

#                 with col1:
#                     st.markdown("#### Attack Type Distribution")
#                     plt.figure(figsize=(6, 4))
#                     sns.countplot(x="Attack_Type", data=df, palette="Set2")
#                     plt.xticks(rotation=45)
#                     st.pyplot(plt.gcf())
#                     plt.clf()

#                 with col2:
#                     st.markdown("#### Attack Type Pie Chart")
#                     pie_data = df["Attack_Type"].value_counts()
#                     fig, ax = plt.subplots()
#                     ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
#                     ax.axis('equal')
#                     st.pyplot(fig)

#                 # Correlation heatmap
#                 st.subheader("📌 Feature Correlation Heatmap")
#                 plt.figure(figsize=(10, 6))
#                 sns.heatmap(df.drop(columns=["Attack_Type"]).corr(), annot=True, cmap="coolwarm")
#                 st.pyplot(plt.gcf())
#                 plt.clf()

#             except Exception as e:
#                 st.error(f"❌ Error during classification: {e}")
# else:
#     st.info("Please upload a CSV file to get started.")
