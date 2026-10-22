import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel
from pyspark.ml.functions import vector_to_array

st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦", layout="wide")

@st.cache_resource
def load_spark_and_models():
    spark = (
        SparkSession.builder
        .appName("BankDepositSubscriptionPrediction")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    root = "models"
    preprocessing = PipelineModel.load(f"{root}/preprocessing_model")
    lr_model = LogisticRegressionModel.load(f"{root}/logistic_regression_model")
    dt_model = DecisionTreeClassificationModel.load(f"{root}/decision_tree_model")
    return spark, preprocessing, lr_model, dt_model

st.title("🏦 Bank Deposit Subscription Predictor")
st.write(
    "Estimate whether a bank customer may subscribe to a term deposit using "
    "the PySpark MLlib models trained for this project."
)
st.info(
    "Educational project demonstration. Predictions are estimates, not financial advice. "
    "The models were trained on historical Bank Marketing data."
)

with st.sidebar:
    st.header("Customer details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services", "student",
        "technician", "unemployed", "unknown"])
    marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education", ["secondary", "tertiary", "primary", "unknown"])
    default = st.selectbox("Has credit in default?", ["no", "yes", "unknown"])
    balance = st.number_input("Average yearly balance (€)", min_value=-100000, max_value=200000, value=1000)
    housing = st.selectbox("Has housing loan?", ["no", "yes", "unknown"])
    loan = st.selectbox("Has personal loan?", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])
    day = st.number_input("Last contact day of month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last contact month", ["jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Contacts during this campaign", min_value=1, max_value=100, value=1)
    pdays = st.number_input("Days since previous campaign contact (-1 = not previously contacted)",
        min_value=-1, max_value=1000, value=-1)
    previous = st.number_input("Contacts before this campaign", min_value=0, max_value=500, value=0)
    poutcome = st.selectbox("Previous campaign outcome", ["unknown", "failure", "other", "success"])

model_choice = st.radio("Choose prediction model", ["Logistic Regression", "Decision Tree"], horizontal=True)

if st.button("Predict subscription likelihood", type="primary", use_container_width=True):
    try:
        spark, preprocessing, lr_model, dt_model = load_spark_and_models()
        customer = pd.DataFrame([{
            "age": int(age),
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": int(balance),
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": int(day),
            "month": month,
            "campaign": int(campaign),
            "pdays": int(pdays),
            "previous": int(previous),
            "poutcome": poutcome,
            "label": 0.0
        }])
        customer_spark = spark.createDataFrame(customer)
        prepared = preprocessing.transform(customer_spark)
        model = lr_model if model_choice == "Logistic Regression" else dt_model
        prediction = model.transform(prepared).withColumn(
            "probability_array", vector_to_array("probability")
        ).first()

        predicted_class = int(prediction["prediction"])
        probability_yes = float(prediction["probability_array"][1])

        left, right = st.columns(2)
        with left:
            st.metric("Estimated subscription probability", f"{probability_yes:.1%}")
            st.progress(min(max(probability_yes, 0.0), 1.0))
        with right:
            if predicted_class == 1:
                st.success("Prediction: Likely to subscribe (yes)")
            else:
                st.warning("Prediction: Unlikely to subscribe (no)")
        st.caption(f"Model used: {model_choice}. The class decision uses the trained model's default threshold.")
    except Exception as exc:
        st.error("Prediction failed. Check that the saved models are present and compatible with the installed Spark version.")
        st.exception(exc)

st.divider()
st.subheader("Model evaluation on the held-out test set")
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Subscriber precision", "Subscriber recall", "ROC-AUC"],
    "Logistic Regression": ["89.69%", "62.79%", "18.75%", "0.7761"],
    "Decision Tree": ["89.45%", "58.84%", "18.15%", "0.6456"]
})
st.dataframe(metrics, use_container_width=True, hide_index=True)
st.caption(
    "Test-set metrics from the original Colab experiment. Subscriber recall is low for both models, "
    "so many actual subscribers are missed at the default classification threshold."
)
st.markdown("**Project:** Bank Deposit Subscription Prediction Using PySpark and Spark MLlib")
