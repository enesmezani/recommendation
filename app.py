import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Function to clean text data
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

# Check if precomputed files exist
if os.path.exists('movies_list.pkl') and os.path.exists('similarity.pkl'):
    movies = pickle.load(open('movies_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
else:
    # Load and preprocess data
    movies = pd.read_csv('movies.csv')
    movies = movies[['id', 'title', 'overview', 'genre']]
    movies['tags'] = movies['overview'] + ' ' + movies['genre']
    new_data = movies.drop(columns=['overview', 'genre'])

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Apply the clean_text function
    new_data['tags_clean'] = new_data['tags'].apply(clean_text)

    # Vectorize the text data using TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    vector = tfidf.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()

    # Calculate cosine similarity
    similarity = cosine_similarity(vector)

    # Serialize the data
    pickle.dump(new_data, open('movies_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Define a function to recommend the top 5 similar movies
def recommend(title, data, similarity_matrix):
    if title not in data['title'].values:
        return ["Movie not found."]
    index = data[data['title'] == title].index[0]
    distance = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda vector: vector[1])
    recommendations = []
    for i in distance[1:10]:
        recommendations.append(data.iloc[i[0]].title)
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_title = request.form['movie_title']
    recommendations = recommend(movie_title, movies, similarity)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
