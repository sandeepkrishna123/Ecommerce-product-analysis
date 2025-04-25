import os
import json
import re
import nltk
import gensim.downloader as api
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)


class DataPreprocessor:
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Load Sentiment Dictionary
        sentiment_dict_path = "./data/preprocess_pkl/sentiment_dict.json"
        with open(sentiment_dict_path, "r") as json_file:
            self.sentiment_dict = json.load(json_file)

    def clean_text(self, text):
        """Lowercase, remove special characters, and stopwords."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
        text = " ".join(
            [word for word in text.split() if word not in self.stop_words]
        )  # Remove stopwords
        return text

    def tokenize_text(self, text):
        """Tokenize text using NLTK."""
        return word_tokenize(text)

    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens using WordNet lemmatizer."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def compute_sentiment_score(self, text):
        """Calculate sentiment score based on preloaded sentiment dictionary."""
        words = word_tokenize(text.lower())  # Tokenize and lowercase
        score = sum(
            self.sentiment_dict.get(word, 0) for word in words
        )  # Sum word sentiment scores
        return score

    def get_review_embedding(self, text):
        """Convert review text into Word2Vec embeddings."""
        words = word_tokenize(text.lower())
        embeddings = [
            self.word2vec_model[word] for word in words if word in self.word2vec_model
        ]

        if embeddings:
            return np.mean(
                embeddings, axis=0
            )  # Average word embeddings for sentence representation
        else:
            return np.zeros(
                self.word2vec_model.vector_size
            )  # Return zero vector if no words found

    def preprocess_dataframe(self, df):
        """Full preprocessing pipeline for DataFrame."""
        # Rename columns
        df.rename(
            columns={
                "reviews.text": "reviews_text",
            },
            inplace=True,
        )

        if "sentiment" in df.columns:
            df.drop(["sentiment"], axis=1, inplace=True)
        # Drop missing values in reviews_text
        df.dropna(subset=["reviews_text"], inplace=True)

        # Apply text cleaning
        df["cleaned_text"] = df["reviews_text"].apply(self.clean_text)

        # Compute Sentiment Scores
        df["sentiment_score"] = df["cleaned_text"].apply(self.compute_sentiment_score)

        # Compute Word2Vec Embeddings
        df["review_embedding"] = df["cleaned_text"].apply(self.get_review_embedding)

        # Combine features into final matrix
        X = np.hstack(
            (
                np.stack(df["review_embedding"].values),
                df["sentiment_score"].values.reshape(-1, 1),
            )
        )

        return df, X

    def preprocess_text_aspect(self, review_text):
        """Preprocess a single review text and return both DataFrame & reshaped feature matrix."""
        # Apply text cleaning
        cleaned_text = self.clean_text(review_text)

        # Compute Sentiment Score
        sentiment_score = self.compute_sentiment_score(cleaned_text)

        # Compute Word2Vec Embedding
        review_embedding = self.get_review_embedding(cleaned_text)

        # Create DataFrame with processed features
        processed_df = pd.DataFrame(
            {
                "review_text": [review_text],
                "sentiment_score": [sentiment_score],
                "review_embedding": [review_embedding],
            }
        )

        # Combine embedding and sentiment score
        feature_vector = np.hstack((review_embedding, np.array([sentiment_score])))

        # Reshape feature vector for model input
        X_reshaped = feature_vector.reshape(1, -1)

        return processed_df, X_reshaped


# ============================ Usage ============================
if __name__ == "__main__":
    word2vec_model = api.load("word2vec-google-news-300")
    preprocessor = DataPreprocessor(word2vec_model)
    df = pd.read_csv("./data/dataset/test_data.csv")
    processed_df, X_train = preprocessor.preprocess_dataframe(df)
    print(processed_df.head(5))
    processed_df.to_csv("./data/dataset/preprocessed_df.csv")
    print("Preprocessing completed successfully.")
    print(X_train.shape)
    sample_review = "The product quality is excellent, but the delivery was slow."
    processed_df, X_review = preprocessor.preprocess_text_aspect(sample_review)

    print("📌 Processed DataFrame:")
    print(processed_df)

    print("\n📌 Reshaped Feature Vector:")
    print(X_review.shape)
