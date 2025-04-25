import streamlit as st
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.visualizations import DataVisualizer
from src.models import ModelLoader
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import gensim.downloader as api
import io
import os
import warnings

import nltk



warnings.filterwarnings("ignore")
nltk.download('punkt_tab')


@st.cache_resource
def load_google_news_encoder():
    word2vec_model = api.load("word2vec-google-news-300")
    return word2vec_model


def convert_df_to_csv(df):
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="styled-title">Exploring E-Commerce Product Experience Based on Fusion Sentiment Analysis Method</h1>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None

    if "pre_df" not in st.session_state:
        st.session_state.pre_df = None

    if "data" not in st.session_state:
        st.session_state.data = None

    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state["page"] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state["page"] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state["page"] = "Sample Training Data"

    if st.sidebar.button("Know About Data"):
        st.session_state["page"] = "Know About Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state["page"] = "Data Preprocessing"

    if st.sidebar.button("Exploratory Data Analysis"):
        st.session_state["page"] = "Exploratory Data Analysis"

    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state["page"] = "Machine Learning Models Used"

    if st.sidebar.button("Upload Test Data"):
        st.session_state["page"] = "Upload Test Data"

    if st.sidebar.button("Model Predictions"):
        st.session_state["page"] = "Model Predictions"

    ################################################################################################################

    if st.session_state["page"] == "Problem Statement":
        st.markdown(DataContent.problem_statement)

    elif st.session_state["page"] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state["page"] == "Sample Training Data":
        st.markdown("## 📊 Training Data Preview")
        st.write("🔍 Below is an **interactive table** displaying the sample data:")
        with st.spinner("⏳ Please wait... Loading training data!"):
            file_path = r".\data\dataset\train_data.csv"
            st.session_state.df = pd.read_csv(file_path)
        st.success("✅ Training data loaded successfully!")

        # Display the table
        data_table = DataTable(df=st.session_state.df.head(100))
        data_table.display_table()

    elif st.session_state["page"] == "Know About Data":
        file_path = r".\data\dataset\train_data.csv"
        report_path = "ydata_profiling_report.html"

        st.session_state.df = pd.read_csv(file_path)
        st.header("📊 Data Information")

        # ✅ Check if the report file already exists
        if os.path.exists(report_path):
            pass
        else:
            with st.status(
                "⏳ Generating Overall Data Profile Report...", expanded=True
            ) as status:
                profile = ProfileReport(st.session_state.df, explorative=True)
                profile.to_file(report_path)
                st.session_state["profile_report_generated"] = True  # Mark as generated
                status.update(
                    label="✅ Report Generated Successfully!", state="complete"
                )

        # ✅ Load and display the report
        try:
            with st.status(
                "⏳ Generating Overall Data Profile Report...", expanded=True
            ) as status:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_html = f.read()
                html(report_html, height=1000, width=800, scrolling=True)

        except FileNotFoundError:
            st.error("❌ Report file not found. Please try again.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    elif st.session_state["page"] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
        pre_df_file = r"./data/dataset/preprocessed_df.csv"
        st.session_state.pre_df = pd.read_csv(pre_df_file)
        st.write("### Preprocessed Data Preview (First 15 Rows)")
        data_table = DataTable(df=st.session_state.pre_df.head(15))
        data_table.display_table()

    elif st.session_state["page"] == "Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")

        file_path2 = "./data/dataset/preprocessed_df.csv"
        st.session_state.pre_df = pd.read_csv(file_path2)

        file_path = "./data/dataset/train_data.csv"
        df = pd.read_csv(file_path)
        visualizer = DataVisualizer()

        plot_type = st.selectbox(
            "Select Visualization",
            [
                "Word Cloud",
                "Target Count Plot",
                "Review Length Distribution",
                "Sentiment Dictionary Contributions",
            ],
        )

        if plot_type == "Word Cloud":
            with st.spinner("Generating Word Cloud..."):
                visualizer.plot_wordcloud(st.session_state.pre_df)

        elif plot_type == "Target Count Plot":
            with st.spinner("Generating Target Count Plot..."):
                fig = visualizer.plot_target_distribution(df)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Review Length Distribution":
            with st.spinner("Generating Review Length Distribution..."):
                fig = visualizer.plot_review_length_distribution(
                    st.session_state.pre_df
                )
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Sentiment Dictionary Contributions":
            with st.spinner("Generating Sentiment Dictionary Contribution Plot..."):
                fig = visualizer.plot_sentiment_dict_contributions()
                st.plotly_chart(fig, use_container_width=True)

    elif st.session_state["page"] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)
        st.markdown("""### 📌 After Training Above All Models here are the Metrics:""")
        df_metrics = pd.read_csv(r".\data\dataset\model_results.csv")
        data_table = DataTable(df=df_metrics)
        data_table.display_table()
        st.markdown(DataContent.best_model)

    elif st.session_state["page"] == "Upload Test Data":
        st.header("Upload or Enter Test Data")

        # Radio button for selecting input type
        input_method = st.radio(
            "Choose Input Method:", ["Upload File", "Enter Manually"]
        )

        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

            if uploaded_file is not None:
                try:
                    with st.spinner(
                        "⏳ Please wait... Data is being uploaded and processed!"
                    ):
                        if uploaded_file.name.endswith(".csv"):
                            data = pd.read_csv(uploaded_file)
                        else:
                            data = pd.read_excel(uploaded_file)

                        st.session_state.data = data

                        # Load BERT model & tokenizer
                        model = load_google_news_encoder()
                        preprocessor = DataPreprocessor(model)

                        # Preprocess the uploaded file
                        preprocessed, st.session_state.preprocessed = (
                            preprocessor.preprocess_dataframe(data)
                        )

                        st.success("✅ Data uploaded and preprocessed successfully!")

                    # Display first 15 rows of preprocessed data
                    data_table = DataTable(df=preprocessed.head(15))
                    data_table.display_table()

                    # Download button for preprocessed data
                    csv_data = convert_df_to_csv(preprocessed)
                    st.download_button(
                        label="📥 Download Full Preprocessed Data",
                        data=csv_data,
                        file_name="preprocessed_test_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        elif input_method == "Enter Manually":
            st.subheader("🔹 Enter Review & Select Aspect Category")

            # User inputs review
            product_name = st.text_input("📌 Product Name:")
            review_text = st.text_area("📝 Write your review here:")

            # Submit button for manual entry
            if st.button("Submit Review"):
                if not review_text.strip():
                    st.warning("⚠️ Please enter a review before submitting!")
                else:
                    with st.spinner("⏳ Processing your review..."):
                        # Load BERT model & tokenizer
                        model = load_google_news_encoder()
                        preprocessor = DataPreprocessor(model)

                        # Preprocess the manually entered text
                        preprocessed, st.session_state.preprocessed = (
                            preprocessor.preprocess_text_aspect(review_text)
                        )
                        st.success("✅ Review processed successfully!")

                    st.write("### Processed Review Embeddings:")
                    data_table = DataTable(df=preprocessed)
                    data_table.display_table()
                    # st.write(preprocessed)

    elif st.session_state["page"] == "Model Predictions":
        if (
            "preprocessed" in st.session_state
            and st.session_state.preprocessed is not None
        ):
            st.header("📊 Sentiment Analysis Predictions")

            loader = ModelLoader()

            # Dropdown for model selection
            model_options = {
                "Fusion Based Logistic Regression": "LogisticRegression",
                "Fusion Based SVM": "SVM",
                "Fusion Based KNN": "KNN",
            }
            model_type = st.selectbox("📌 Select Model:", list(model_options.keys()))

            if st.button("🚀 Load Model & Predict"):
                with st.spinner("⏳ Loading model and making predictions..."):
                    model_filename = model_options[model_type]
                    X_test = st.session_state.preprocessed  # Preprocessed embeddings

                    predictions, metrics = loader.predict_and_evaluate(
                        model_filename, X_test
                    )

                    if predictions is not None:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("🔍 Sample Predictions")
                            data_table = DataTable(df=predictions.head(10))
                            data_table.display_table_new()

                            # Allow user to download full predictions
                            csv = predictions.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Full Predictions",
                                data=csv,
                                file_name="sentiment_predictions.csv",
                                mime="text/csv",
                                key="download_button",
                            )

                        with col2:
                            st.subheader("📈 Model Metrics")
                            if metrics is not None:
                                metrics_transposed = metrics.T.reset_index()
                                metrics_transposed.columns = ["Metric", "Value"]
                                data_table = DataTable(df=metrics_transposed)
                                data_table.display_table_new()
                            else:
                                st.warning(
                                    f"⚠️ No stored metrics found for {model_type}."
                                )
                    else:
                        st.error("❌ Failed to load model or make predictions.")
        else:
            st.warning("⚠️ Please upload and preprocess data first!")


if __name__ == "__main__":
    main()
