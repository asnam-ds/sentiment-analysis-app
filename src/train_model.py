import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("Loading dataset...")
df = pd.read_csv('data/imdb_reviews.csv')
print(f"Total rows: {len(df)}")
print(f"Sentiment counts:\n{df['sentiment'].value_counts()}")

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens
               if w not in stop_words and len(w) > 2]
    return ' '.join(cleaned)

print("Cleaning text... please wait 2-3 minutes")
df['clean_review'] = df['review'].apply(clean_text)
print("Cleaning done!")
df['label'] = (df['sentiment'] == 'positive').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'],
    test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"ACCURACY: {accuracy*100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/sentiment_model.pkl','wb'))
pickle.dump(tfidf, open('model/tfidf_vectorizer.pkl','wb'))
print("Model saved! All done.")