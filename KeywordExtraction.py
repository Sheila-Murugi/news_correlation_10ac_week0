import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from CSV files
data = pd.read_csv('C:/Users/HP/Desktop/KARUGIII/data.csv/rating.csv')
domains_location = pd.read_csv('C:/Users/HP/Desktop/KARUGIII/domains_location.csv')

# Extract headlines and news bodies
headlines = data['title'].tolist()
news_bodies = data['content'].tolist()

# Combine headlines and news bodies
text_data = [f"{headline} {news_body}" for headline, news_body in zip(headlines, news_bodies)]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
# Perform analysis
# For example, calculate the top N most important words using TF-IDF scores

# Get the feature names (words) from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get the TF-IDF scores for each feature (word)
tfidf_scores = X.toarray()

# Calculate the mean TF-IDF score for each word across all documents
mean_tfidf_scores = tfidf_scores.mean(axis=0)

# Sort the words based on their mean TF-IDF scores in descending order
sorted_words_indices = mean_tfidf_scores.argsort()[::-1]

# Print the top N most important words and their TF-IDF scores
top_n = 10
print(f"Top {top_n} most important words:")
for i in range(top_n):
    word_index = sorted_words_indices[i]
    word = feature_names[word_index]
    tfidf_score = mean_tfidf_scores[word_index]
    print(f"{word}: {tfidf_score}")
