import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder


class DataContent:
    """Class to store project markdown descriptions."""

    problem_statement = """
        ### 🎓 **Problem Statement**  
        With the **rapid expansion of e-commerce**, online product reviews have become a crucial source of consumer insights, influencing purchasing decisions and business strategies. However, extracting meaningful sentiment information from massive online reviews poses several challenges:  

        🔹 **Difficulty in accurately classifying sentiment polarity**, leading to misinterpretation of customer opinions.  
        🔹 **Limited ability to capture aspect-based sentiments**, making it challenging to analyze reviews across different product features.  
        🔹 **Loss of emotional information**, as traditional machine learning models fail to handle **contextual sentiment variations** effectively.  

        ---

        #### 🎯 **Project Objective**  
        This research aims to develop a **fusion sentiment analysis model** that integrates **textual analysis techniques with machine learning algorithms** to accurately mine consumer experiences from online product reviews. The proposed approach includes:  

        ✅ **Sentiment Dictionary-Based Feature Extraction** – Captures emotional cues and enhances the precision of sentiment analysis.  
        ✅ **Support Vector Machine (SVM) Classification** – Effectively identifies sentiment polarity in reviews.  
      
        ✅ **Semantic Dictionary Expansion** – Ensures comprehensive sentiment coverage by extending the dictionary using **semantic similarity**.  
        ✅ **Weighting-Based Sentiment Contribution Measurement** – Assigns different weights to words based on their **sentiment strength**, improving feature extraction quality.  

        This approach enhances the accuracy and reliability of sentiment classification, enabling businesses to **better understand customer experiences and optimize product strategies**.  

        ---

        #### 🧠 **Machine Learning Models Used**  
        The proposed **fusion sentiment analysis model** incorporates:  

        🔹 **Sentiment Dictionary + Semantic Expansion** – Extracts sentiment-rich features while avoiding loss of key emotional information.  
        🔹 **Support Vector Machine (SVM)** – Efficiently classifies sentiment polarity from textual data.   
        🔹 **Word Embeddings (Word2Vec, TF-IDF)** – Improves sentiment feature extraction by representing text numerically.  

        ---

        #### 🔍 **Key Challenges Addressed**  
        ✅ **Handling of Subjective Sentiments** – Improves sentiment classification by incorporating **dictionary-based** and **ML-based** techniques.  
        ✅ **Aspect-Based Sentiment Mining** – Extracts opinions on different product features like **quality, usability, and service**.  
        ✅ **Mitigating Data Imbalance in Sentiment Classes** – Uses advanced **feature weighting** and **semantic similarity expansion**.  
        ✅ **Scalability & Real-World Application** – The model is adaptable for **e-commerce platforms, online reviews, and consumer feedback analysis**.  

        ---

        #### 🌐 **Real-World Applications & End Users**  
        🔹 **E-commerce Platforms (Amazon, eBay, Flipkart)** – Analyze customer reviews to improve product recommendations.  
        🔹 **Product Managers & Marketers** – Gain insights into consumer preferences for product development and marketing strategies.  
        🔹 **Consumers** – Understand product sentiment trends before making purchasing decisions.  
        🔹 **Retail Analytics & Market Research** – Track customer feedback to refine pricing, inventory, and branding strategies.  

        🏆 **This research introduces a robust sentiment analysis model that enhances the accuracy of e-commerce product review analysis, improves customer insights, and optimizes business strategies.**  
        """

    project_data_details = """
        ### 📊 **Project Data Description**  

        #### 🔍 **Context**  
        The **e-commerce product review sentiment analysis dataset** is a collection of customer feedback gathered from Amazon. This dataset provides valuable insights into **consumer experiences, product quality, and customer satisfaction** across various product categories.  

        The dataset enables **sentiment analysis** to determine whether a product review expresses a **positive or negative opinion**, helping businesses enhance their **product offerings, customer service, and marketing strategies**.  

        ---  

        #### 📂 **Content**  
        The dataset consists of **4,000 customer reviews** with the following key features:  

        ✅ **name** – The name of the product reviewed.  
        ✅ **brand** – The brand associated with the product.  
        ✅ **categories** – The category under which the product falls (e.g., electronics, clothing, home appliances).  
        ✅ **primaryCategories** – The broader category of the product (e.g., Books, Electronics, Fashion).  
        ✅ **reviews_date** – The date when the review was posted.  
        ✅ **reviews_text** – The full text of the customer review.  
        ✅ **reviews_title** – The title of the review provided by the customer.  
        ✅ **sentiment** – The sentiment polarity of the review (**0: Negative, 1: Positive**).  

        **Final Dataset Overview:**  
        This dataset contains **4,000 product reviews**, where **sentiments are explicitly labeled** as **positive or negative**, making it suitable for sentiment classification using **machine learning and deep learning techniques**.  

        ---  

        #### 📊 **Key Features Used**  
        This dataset captures various elements essential for **sentiment analysis** in e-commerce product reviews:  

        ✅ **Text-Based Features**  
        - **Review Text (reviews_text)** – The detailed customer feedback.  
        - **Review Title (reviews_title)** – A short summary of the customer's review.  

        ✅ **Product-Specific Features**  
        - **Product Name (name)** – Identifies the reviewed product.  
        - **Brand (brand)** – Specifies the manufacturer or seller.  
        - **Category Information (categories, primaryCategories)** – Helps in aspect-based sentiment analysis.  

        ✅ **Sentiment Classification**  
        - **Sentiment Label (sentiment)** – Defines the polarity of the sentiment (**0: Negative, 1: Positive**).  

        ✅ **Metadata**  
        - **Review Date (reviews_date)** – Provides a timeline for sentiment trends.  

        ---  

        #### 🎯 **Alignment with Methodology**  
        This dataset enables the application of **Machine Learning & NLP Techniques** for sentiment classification:  
        🔹 **Feature Extraction** – Using **Word2Vec and TF-IDF** to transform text into meaningful feature representations.  
        🔹 **Sentiment Classification** – Leveraging **Support Vector Machine (SVM) and Logistic Regression** for sentiment prediction.  
        🔹 **Aspect-Based Analysis** – Categorizing sentiment trends across different product categories and brands.  

        ---  

        🏆 **This dataset serves as a foundation for AI-driven sentiment analysis, allowing businesses to gain deeper insights into customer feedback and improve product offerings.**   
        
        """

    Data_preprocessing = """
        ### 📌 **Data Preprocessing**  

        #### 🛠 **1️⃣ Importance of Data Preprocessing in Sentiment Analysis**  
        Data preprocessing is a **fundamental step** in sentiment analysis that ensures data quality and enhances model performance. In this project, we are implementing a **fusion-based sentiment analysis approach**, combining **dictionary-based sentiment scoring and word embeddings** to improve sentiment classification accuracy.  

        This preprocessing pipeline is specifically designed to align with the **proposed fusion-based framework**, which integrates both **traditional sentiment analysis techniques (sentiment dictionary, TF-IDF-based feature extraction)** and **deep learning approaches (Word2Vec embeddings, machine learning models)**. By leveraging both methodologies, we ensure that the model effectively captures **explicit sentiment expressions** (from the sentiment dictionary) and **contextual semantic relationships** (from word embeddings). This hybrid approach enhances sentiment classification accuracy and adaptability across different product categories.  

        ---  

        #### 🛠 **2️⃣ Initial Data Checks & Basic Cleaning**  
        Before applying advanced NLP techniques, we performed several **data validation and cleaning steps**:  

        ✅ **Type Conversion & Schema Validation**  
        - Ensured all features had the correct data types for consistency.  
        - Standardized column names for ease of access in future processing.  

        ✅ **Handling Missing & Unlabeled Data**  
        - Removed records where **reviews were empty** (since sentiment analysis relies on textual data).  
        - Dropped records **without sentiment labels**, ensuring all data points had valid sentiment annotations.  

        ✅ **Feature Renaming for Standardization**  
        - Renamed columns such as **reviews.text → reviews_text**, maintaining uniformity for further processing.  

        ---  

        #### 🛠 **3️⃣ NLP-Based Text Preprocessing**  
        To refine the text for sentiment classification, we applied multiple **linguistic preprocessing techniques**:  

        ✅ **Text Cleaning & Stopword Removal**  
        - Converted text to **lowercase** for uniformity.  
        - Removed **special characters and punctuation** to reduce noise.  
        - Eliminated **stopwords** that do not contribute to sentiment meaning.  

        ✅ **Tokenization & Lemmatization**  
        - Tokenized reviews into **individual words** for further processing.  
        - Applied **WordNet Lemmatization** to convert words into their root forms, reducing vocabulary size.  

        ✅ **TF-IDF for Sentiment Dictionary Expansion**  
        - Applied **TF-IDF (Term Frequency-Inverse Document Frequency)** to identify important missing sentiment words.  
        - Extracted high-impact words that were not in the original **sentiment dictionary** and assigned scores based on their importance in the dataset.  

        ✅ **Sentiment Score Computation**  
        - Calculated **sentiment scores** for each review using the **sentiment dictionary**.  
        - Aggregated the sentiment contributions of all words in a review to derive a **final sentiment score** for that review.  

        ---  

        #### 🛠 **4️⃣ Feature Engineering: Word2Vec Embeddings**  
        To capture deep **semantic relationships** within the text, we used **pretrained Word2Vec embeddings**:  

        ✅ **Loaded Pretrained Word2Vec Model**  
        - Used **Google News Word2Vec (300-dimensional embeddings)** to convert words into numerical vectors.  

        ✅ **Review Embeddings Computation**  
        - Transformed each review into **Word2Vec representations** by averaging the embeddings of all words in the review.  

        ✅ **Fusion of Sentiment Scores & Word Embeddings**  
        - Combined **Word2Vec review embeddings** with **sentiment scores** to create an enhanced feature representation for the model.  

        ---  

        #### 🛠 **5️⃣ Sentiment Label Mapping & Validation Set Processing**  
        ✅ **Standardized Sentiment Labels**  
        - Mapped sentiment labels to binary values:  
        - **Positive Sentiment → 1**  
        - **Negative Sentiment → 0**  

        ✅ **Validation Data Processing**  
        - Applied **identical preprocessing steps** to the validation dataset, ensuring consistency between train and validation data.  
        - No **train-test split** was required since we had separate files for training and validation.  

        ---  

        #### 🛠 **6️⃣ Data Reshaping for Model Input**  
        To prepare the data for model training, we transformed the dataset into a format suitable for **deep learning models**:  

        ✅ **Feature Matrix Construction**  
        - Combined **Word2Vec embeddings** and **sentiment scores** into a structured feature matrix.  

        ✅ **Reshaping for Model Compatibility**  
        - Adjusted data dimensions to ensure compatibility with the deep learning model’s expected input format.  

        ---  

        🏆 **This structured preprocessing pipeline effectively prepares the dataset for fusion-based sentiment analysis, leveraging both sentiment dictionary-based scoring and deep learning-based word embeddings to enhance sentiment classification accuracy. By combining these two approaches, the model benefits from both explicit sentiment features and contextual semantic representations, m

        """

    ml_models = """
        ### 🚀 **Fusion-Based Machine Learning Models for Sentiment Analysis**  

        Traditional machine learning models struggle with **contextual sentiment understanding**. However, our approach **enhances sentiment classification** by integrating:  
        🔹 **Precomputed Sentiment Scores** – Derived from a custom **sentiment dictionary-based approach**.  
        🔹 **Word2Vec Embeddings** – Capturing deep semantic relationships between words.  
        🔹 **Feature Fusion** – Combining both **sentiment scores & embeddings** before passing them to the model.  

        This fusion-based methodology improves accuracy and model interpretability, making the classification **more robust and context-aware**.  

        ---

        ### 1️⃣ **Logistic Regression (LR)**  
        Logistic Regression is a simple yet powerful algorithm for **binary classification**, making it suitable for **sentiment polarity detection**.  
        It models the probability of a review being **positive (1) or negative (0)** using a **logistic (sigmoid) function**.  

        🔹 **Why Logistic Regression?**  
        - Works well with **high-dimensional feature spaces** like embeddings.  
        - **Interpretable** – Provides insight into which features contribute most to sentiment prediction.  
        - **Handles noise well** – Useful for real-world text data.  

        🔹 **Hyperparameters Used:**  
        - **Solver:** `lbfgs` (efficient optimization for small datasets).  
        - **Max Iterations:** `1000` (ensures proper convergence).  
        - **Class Weight:** `balanced` (adjusts weights to handle class imbalance).  

        🔹 **Performance Metrics:**  
        - **Train Accuracy:** 89.06%  
        - **Test Accuracy:** 89.25%  
        - **Train Precision:** 97.95%  
        - **Test Precision:** 99.64%  
        - **Train Recall:** 89.06%  
        - **Test Recall:** 89.29%  
        - **Train F1-Score:** 92.53%  
        - **Test F1-Score:** 94.18%  

        ---

        ### 2️⃣ **Support Vector Machine (SVM)**  
        SVM is a **powerful supervised learning algorithm** that finds the best **hyperplane** to separate positive and negative sentiment reviews.  
        By leveraging **support vectors**, it maximizes the margin between different sentiment classes, making it highly effective for text classification tasks.  

        🔹 **Why SVM?**  
        - Works well with **small to medium-sized datasets**.  
        - **Handles high-dimensional data** (like Word2Vec embeddings) effectively.  
        - **Robust against overfitting** when used with proper regularization.  

        🔹 **Hyperparameters Used:**  
        - **Kernel:** `linear` (ensures interpretability and efficiency).  
        - **Class Weight:** `balanced` (handles class distribution).  

        🔹 **Performance Metrics:**  
        - **Train Accuracy:** 90.83%  
        - **Test Accuracy:** 89.77%  
        - **Train Precision:** 98.11%  
        - **Test Precision:** 99.64%  
        - **Train Recall:** 90.83%  
        - **Test Recall:** 89.83%  
        - **Train F1-Score:** 93.62%  
        - **Test F1-Score:** 94.48%  

        ---

        ### 3️⃣ **K-Nearest Neighbors (KNN)**  
        KNN is a **non-parametric, instance-based learning algorithm** that classifies sentiment by finding the **k-nearest neighbors** of a given review.  
        It calculates distances between data points and assigns the most common class among its nearest neighbors.  

        🔹 **Why KNN?**  
        - **Highly adaptable** – Does not assume any data distribution.  
        - **Works well with fusion-based feature representations** (sentiment scores + embeddings).  
        - **Effectively captures local patterns** in the dataset.  

        🔹 **Hyperparameters Used:**  
        - **Number of Neighbors (K):** `5` (optimal for balanced accuracy).  
        - **Weights:** `distance` (closer neighbors have more influence).  

        🔹 **Performance Metrics:**  
        - **Train Accuracy:** 100%  
        - **Test Accuracy:** 98.12%  
        - **Train Precision:** 100%  
        - **Test Precision:** 98.21%  
        - **Train Recall:** 100%  
        - **Test Recall:** 99.89%  
        - **Train F1-Score:** 100%  
        - **Test F1-Score:** 99.04%  
        
        """

    best_model = """
        ### ✅ **Best Model: K-Nearest Neighbors (KNN)**  

        Among the three models evaluated, **KNN demonstrated the highest performance** on test data, making it the optimal choice for **fusion-based sentiment analysis**.  

        ### 🔹 **Why is KNN the Best Model?**  
        - **Achieved the highest test accuracy (98.12%)**, outperforming both Logistic Regression and SVM.  
        - **Perfectly learned the training data (100% accuracy)** while still generalizing well on unseen data.  
        - **Adapted well to the fusion-based approach**, effectively utilizing both **sentiment scores and embeddings**.  
        - **Distance-weighted KNN ensured optimal classification**, allowing closer neighbors to have more influence.  

        ### 📊 **Performance Metrics:**  
        - **Train Accuracy:** 100%  
        - **Test Accuracy:** 98.12%  
        - **Train Precision:** 100%  
        - **Test Precision:** 98.21%  
        - **Train Recall:** 100%  
        - **Test Recall:** 99.89%  
        - **Train F1-Score:** 100%  
        - **Test F1-Score:** 99.04%  

        KNN effectively leverages **both explicit sentiment information (sentiment scores) and contextual word relationships (embeddings)**, making it the best-performing model for our dataset.  
        This confirms that our **fusion-based approach** successfully enhances sentiment classification accuracy, setting a strong foundation for real-world applications in **e-commerce product review analysis**.  
        """


class DataTable:
    """Class to handle dataset loading and displaying with AgGrid."""

    def __init__(self, df):
        self.df = df

    def display_table(self):
        df_preview = self.df

        gb = GridOptionsBuilder.from_dataframe(df_preview)
        gb.configure_default_column(
            groupable=True, value=True, enableRowGroup=True, editable=False
        )

        # Custom Styling
        gb.configure_grid_options(
            rowHeight=40,
            headerHeight=60,
            domLayout="autoHeight",
            suppressHorizontalScroll=True,
            enableSorting=True,
            enableFilter=True,
            rowSelection="multiple",
            suppressColumnVirtualisation=True,  # Prevent stretching
        )

        grid_options = gb.build()
        num_rows = len(df_preview)
        max_table_height = 600  # Adjust this as needed
        final_height = min(num_rows * 90, max_table_height)

        custom_css = {
            ".ag-header": {
                "background-color": "#0047AB",
                "color": "#FFFFFF",
                "font-size": "16px",
                "font-weight": "bold",
                "text-align": "center",
                "border-bottom": "2px solid #CCCCCC",
                "padding": "10px",
            },
            ".ag-header-cell": {
                "background-color": "#0047AB !important",
                "color": "#FFFFFF !important",
                "border": "none",
                "padding": "5px",
                "height": "50px",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
            ".ag-row-odd": {
                "background-color": "#F8F9FA",
            },
            ".ag-row-even": {
                "background-color": "#E9ECEF",
            },
            ".ag-body": {
                "border": "2px solid #CCCCCC",
            },
            ".ag-cell": {
                "font-size": "14px",
                "color": "#333333",
                "border-right": "1px solid #CCCCCC",  # Add grid lines to cells
            },
            ".ag-row-hover": {
                "background-color": "#D1E8FF !important",  # Light blue hover
            },
            ".ag-cell-focus": {
                "border": "none !important",  # Remove black border on focus
            },
            ".ag-body-viewport": {
                "background-color": "#FFFFFF !important",  # White background for empty space
            },
            ".ag-center-cols-viewport": {
                "background-color": "#FFFFFF !important",  # White background for empty space
                "border-right": "1px solid #CCCCCC",  # Extend grid lines to the end
            },
        }

        # Apply AgGrid with Custom Styling
        AgGrid(
            df_preview,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            theme="balham",
            height=final_height,
            custom_css=custom_css,
        )

    def display_table_new(self):
        df_preview = self.df.head(100)

        gb = GridOptionsBuilder.from_dataframe(df_preview)
        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            editable=False,
            flex=1,  # Add flex to distribute width
        )

        # Calculate approximate width needed
        num_rows = len(df_preview)
        num_columns = len(df_preview.columns)
        base_column_width = 300  # Adjust based on your typical content width
        total_columns_width = num_columns * base_column_width
        max_table_height = 500  # Adjust this as needed
        final_height = min(num_rows * 140, max_table_height)

        gb.configure_grid_options(
            rowHeight=40,
            headerHeight=70,
            domLayout="normal",  # Changed from autoHeight
            suppressHorizontalScroll=False,  # Allow horizontal scroll if needed
            enableSorting=True,
            enableFilter=True,
            rowSelection="multiple",
            suppressColumnVirtualisation=True,
            # Set dynamic width based on columns
            gridWidth=total_columns_width if num_columns < 5 else "200%",
        )

        grid_options = gb.build()

        # Update custom CSS to handle dynamic width
        custom_css = {
            ".ag-header": {
                "background-color": "#0047AB",
                "color": "#FFFFFF",
                "font-size": "16px",
                "font-weight": "bold",
                "text-align": "center",
                "border-bottom": "2px solid #CCCCCC",
                "padding": "10px",
            },
            ".ag-header-cell": {
                "background-color": "#0047AB !important",
                "color": "#FFFFFF !important",
                "border": "none",
                "padding": "5px",
                "height": "50px",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
            ".ag-row-odd": {
                "background-color": "#F8F9FA",
            },
            ".ag-row-even": {
                "background-color": "#E9ECEF",
            },
            ".ag-body": {
                "border": "2px solid #CCCCCC",
            },
            ".ag-cell": {
                "font-size": "14px",
                "color": "#333333",
                "border-right": "1px solid #CCCCCC",  # Add grid lines to cells
            },
            ".ag-row-hover": {
                "background-color": "#D1E8FF !important",  # Light blue hover
            },
            ".ag-cell-focus": {
                "border": "none !important",  # Remove black border on focus
            },
            ".ag-body-viewport": {
                "background-color": "#FFFFFF !important",  # White background for empty space
            },
            ".ag-center-cols-viewport": {
                "background-color": "#FFFFFF !important",  # White background for empty space
                "border-right": "1px solid #CCCCCC",  # Extend grid lines to the end
            },
        }

        # Apply AgGrid with dynamic width
        AgGrid(
            df_preview,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            theme="balham",
            height=final_height,
            custom_css=custom_css,
            width=total_columns_width if num_columns < 10 else "100%",
            fit_columns_on_grid_load=True,  # Add this to fit columns to content
        )
