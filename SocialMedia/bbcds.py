# Loading Data
import numpy as np
import pandas as pd

import warnings
df=pd.read_csv('bbc.csv')
df.head()
print(df.shape)
df.info()
df.isna().sum()

# ANALYZING TEXT COLUMN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

# Tokenize the text column into sentences
df['sentences'] = df['text'].apply(sent_tokenize)

# Tokenize the sentences into words
df['words'] = df['sentences'].apply(lambda x: [word_tokenize(sent) for sent in x])

import string
# Remove punctuation marks
punctuations = string.punctuation
df['words_no_punct'] = df['words'].apply(lambda x: [[word.translate(str.maketrans('', '', string.punctuation)) for word in sent] for sent in x])

# Remove stop words
stop_words = set(stopwords.words('english'))
df['words_cleaned'] = df['words_no_punct'].apply(lambda x: [[word for word in sent if word.lower() not in stop_words] for sent in x])

# Stem the words
stemmer = PorterStemmer()
df['words_stemmed'] = df['words_cleaned'].apply(lambda x: [[stemmer.stem(word) for word in sent] for sent in x])

# Flatten the list of lists
df['words_flat'] = df['words_stemmed'].apply(lambda x: [item for sublist in x for item in sublist])

# Count word frequencies
word_counts = Counter()
for words in df['words_flat']:
    word_counts.update(words)

# Print the most common words
print("Most common words:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")

# ANALYZING SENTIEMNT SCORE
from nltk.sentiment import SentimentIntensityAnalyzer
# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def calculate_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Apply the sentiment analysis function to the 'text' column
df['sentiment_score'] = df['text'].apply(calculate_sentiment)

# Print the sentiment scores
print(df[['text', 'sentiment_score']])

import matplotlib.pyplot as plt

# Categorize sentiment scores into positive, neutral, and negative
bins = [-1, -0.05, 0.05, 1]
labels = ['Negative', 'Neutral', 'Positive']
df['sentiment_category'] = pd.cut(df['sentiment_score'], bins=bins, labels=labels)

# Count the occurrences of each sentiment category
sentiment_counts = df['sentiment_category'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar')
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Sentiment Categories', fontsize=16)
plt.xticks(rotation=0)

# Add value labels on top of the bars
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v, str(v), fontsize=10, horizontalalignment='center', verticalalignment='bottom')

plt.show()

# ANALYZING ENGAGEMENT LEVELS
# Overall Engagement Levels
print("Overall Engagement Levels:")
print("Total Likes:", df['likes'].sum())
print("Total Comments:", df['comments'].sum())
print("Total Shares:", df['shares'].sum())

import seaborn as sns
# Engagement Distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['likes'], kde=True)
plt.title('Distribution of Likes')

plt.subplot(1, 3, 2)
sns.histplot(df['comments'], kde=True)
plt.title('Distribution of Comments')

plt.subplot(1, 3, 3)
sns.histplot(df['shares'], kde=True)
plt.title('Distribution of Shares')

plt.tight_layout()
plt.show()

# Correlation Analysis
engagement_corr = df[['likes', 'comments', 'shares']].corr()
print("\nCorrelation Matrix:")
print(engagement_corr)

# Engagement by Sentiment
sentiment_engagement = df.groupby('sentiment_category')[['likes', 'comments', 'shares']].sum().reset_index()
print("\nEngagement by Sentiment:")
print(sentiment_engagement)

# Engagement Ratios by Sentiment
engagement_ratios_by_sentiment = df.groupby('sentiment_category')[['likes', 'comments', 'shares']].sum().reset_index()
engagement_ratios_by_sentiment['comments_to_likes'] = engagement_ratios_by_sentiment['comments'] / engagement_ratios_by_sentiment['likes']
engagement_ratios_by_sentiment['shares_to_likes'] = engagement_ratios_by_sentiment['shares'] / engagement_ratios_by_sentiment['likes']

print("\nEngagement Ratios by Sentiment:")
print(engagement_ratios_by_sentiment[['sentiment_category', 'comments_to_likes', 'shares_to_likes']])

# Visualize Engagement Ratios by Sentiment
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='sentiment_category', y='comments_to_likes', data=engagement_ratios_by_sentiment)
plt.title('Comments to Likes Ratio by Sentiment')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='sentiment_category', y='shares_to_likes', data=engagement_ratios_by_sentiment)
plt.title('Shares to Likes Ratio by Sentiment')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

