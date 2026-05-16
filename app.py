import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

@st.cache_resource
def load_model():
    model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
    return model, tfidf

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

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Sentiment Analysis App")
st.markdown("**Type a movie review and I'll tell you if it's Positive or Negative**")
st.markdown("---")

model, tfidf = load_model()

col1, col2, col3 = st.columns(3)
if col1.button("😊 Positive Example"):
    st.session_state.text = "This movie was absolutely fantastic! Best film I have seen in years."
if col2.button("😞 Negative Example"):
    st.session_state.text = "Terrible movie. Complete waste of time. The plot made no sense."
if col3.button("🔄 Clear"):
    st.session_state.text = ""

review = st.text_area("Enter your review:",
    value=st.session_state.get('text', ''),
    height=150,
    placeholder="Type a movie review here...")

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100

        st.markdown("---")
        if prediction == 1:
            st.success("## ✅ POSITIVE Sentiment")
        else:
            st.error("## ❌ NEGATIVE Sentiment")

        st.markdown(f"**Confidence: {confidence:.1f}%**")
        st.progress(int(confidence))

        col_a, col_b = st.columns(2)
        col_a.metric("😞 Negative", f"{proba[0]*100:.1f}%")
        col_b.metric("😊 Positive", f"{proba[1]*100:.1f}%")

st.markdown("---")
st.caption("Built with Python + Scikit-learn + Streamlit | MSc Applied Data Science")