import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import json


class DataVisualizer:
    def __init__(self):
        self.layout = {
            "template": "plotly_white",
            "plot_bgcolor": "white",
            "paper_bgcolor": "black",
            "font": dict(color="white"),
        }
        sentiment_dict_path = "./data/preprocess_pkl/sentiment_dict.json"
        with open(sentiment_dict_path, "r") as json_file:
            self.sentiment_dict = json.load(json_file)

    def plot_wordcloud(self, data):
        """
        Generate a word cloud for the 'reviews_text' column.
        """
        text = " ".join(data["reviews_text"].astype(str))
        wordcloud = WordCloud(
            width=800, height=400, background_color="black", colormap="viridis"
        ).generate(text)

        # Display word cloud using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    def plot_target_distribution(self, data):
        """
        Generate a count plot for the 'sentiment' column.
        """
        data["sentiment"] = data["sentiment"].map({"Positive": 1, "Negative": 0})

        label_counts = data["sentiment"].value_counts().reset_index()
        label_counts.columns = ["sentiment", "Count"]

        fig = px.bar(
            label_counts,
            x="sentiment",
            y="Count",
            text="Count",
            color="sentiment",
            title="Sentiment Label Distribution",
            labels={"sentiment": "Sentiment Class", "Count": "Number of Reviews"},
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=-45,
            template="plotly_white",
            width=900,
            height=500,
        )

        return fig

    def plot_review_length_distribution(self, data):
        """
        Generate a histogram for the length of reviews.
        """
        data["review_length"] = data["reviews_text"].apply(
            lambda x: len(str(x).split())
        )

        fig = px.histogram(
            data,
            x="review_length",
            nbins=50,
            title="Review Length Distribution",
            labels={"review_length": "Number of Words", "count": "Number of Reviews"},
        )

        fig.update_layout(
            **self.layout,
            width=900,
            height=500,
        )

        return fig

    def plot_sentiment_dict_contributions(self):
        """
        Generate a bar chart of top sentiment words based on sentiment scores.
        """
        # Convert dictionary to DataFrame
        sentiment_df = pd.DataFrame(
            list(self.sentiment_dict.items()), columns=["Word", "Sentiment Score"]
        )

        # Select top positive and negative words
        top_positive = sentiment_df.nlargest(10, "Sentiment Score")
        top_negative = sentiment_df.nsmallest(10, "Sentiment Score")

        # Merge both
        top_words = pd.concat([top_positive, top_negative])

        # Plot
        fig = px.bar(
            top_words,
            x="Word",
            y="Sentiment Score",
            color="Sentiment Score",
            title="Top Sentiment Words Contribution",
            labels={"Sentiment Score": "Sentiment Contribution"},
            text="Sentiment Score",
            color_continuous_scale="RdBu",
        )

        fig.update_layout(width=900, height=500, xaxis_tickangle=-45)
        return fig
