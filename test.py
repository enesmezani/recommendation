import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load and preprocess data
movies = pd.read_csv('movies.csv')
movies = movies[['id', 'title', 'overview', 'genre']]
movies['tags'] = movies['overview'] + movies['genre']
new_data = movies.drop(columns=['overview', 'genre'])

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define a function for cleaning text data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\d]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text

# Apply the clean_text function
new_data['tags_clean'] = new_data['tags'].apply(clean_text)

# Debug: Print the first few cleaned tags
print(new_data['tags_clean'].head())

# Vectorize the text data
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()

# Debug: Check the shape of the vectorized data
print("Vector shape:", vector.shape)

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Debug: Print a portion of the similarity matrix
print("Similarity matrix sample:\n", similarity[:5, :5])

# Define a function to recommend the top 5 similar movies
def recommend(title, data, similarity_matrix):
    if title not in data['title'].values:
        print("Movie not found.")
        return
    index = data[data['title'] == title].index[0]
    distance = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda vector: vector[1])
    # Skip the first movie as it is the movie itself
    print(f"Top 5 recommendations for '{title}':")
    for i in distance[1:6]:  # Get top 5 recommendations excluding the input movie itself
        print(data.iloc[i[0]].title)

# Test the recommend function with a specific movie
recommend("Spirited Away", new_data, similarity)

# Serialize the data
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Print the current working directory
print(os.getcwd())
