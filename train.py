import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = {
    "text": [
        "I am very happy today",
        "This is the best day ever",
        "I feel so sad and lonely",
        "I want to cry",
        "I am very angry right now",
        "This makes me furious",
        "I feel calm and peaceful",
        "Everything feels relaxed",
        "I am stressed and overwhelmed",
        "Too much pressure on me"
    ],
    "emotion": [
        "happy", "happy",
        "sad", "sad",
        "angry", "angry",
        "calm", "calm",
        "stressed", "stressed"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["emotion"]

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained 💗")
