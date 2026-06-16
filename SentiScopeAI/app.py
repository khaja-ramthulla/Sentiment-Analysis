from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import yake

from nltk.tokenize import sent_tokenize

nltk.download("punkt")

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Keyword extractor
keyword_extractor = yake.KeywordExtractor(
    lan="en",
    n=1,
    top=8
)

positive_words = [
    "love",
    "excellent",
    "great",
    "amazing",
    "happy",
    "fantastic",
    "wonderful",
    "best"
]

negative_words = [
    "hate",
    "terrible",
    "awful",
    "bad",
    "horrible",
    "worst",
    "angry",
    "disappointing"
]


def detect_emotion(text):

    text = text.lower()

    positive_score = sum(
        word in text
        for word in positive_words
    )

    negative_score = sum(
        word in text
        for word in negative_words
    )

    if positive_score > negative_score:
        return "😊 Joy"

    elif negative_score > positive_score:
        return "😡 Anger"

    return "😐 Neutral"


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    text = request.form.get(
        "text",
        ""
    )

    if not text.strip():

        return jsonify({
            "error":
            "Please enter text."
        })

    # Main prediction
    text_vector = vectorizer.transform(
        [text]
    )

    probability = model.predict_proba(
    text_vector
    )[0]

    confidence = round(
    max(probability) * 100,
    2
    )

    if confidence < 60:
        prediction = "neutral"
    else:
        prediction = model.predict(
            text_vector
        )[0]
    # Keywords
    keywords = [
        keyword
        for keyword, score
        in keyword_extractor.extract_keywords(text)
    ]

    # Sentence Analysis
    sentences = sent_tokenize(text)

    sentence_results = []

    for sentence in sentences:

        sentence_vector = vectorizer.transform(
            [sentence]
        )

        sentence_prediction = model.predict(
            sentence_vector
        )[0]

        sentence_results.append({
            "sentence": sentence,
            "sentiment": sentence_prediction
        })

    # Stats
    word_count = len(
        text.split()
    )

    reading_time = round(
        word_count / 200,
        2
    )

    emotion = detect_emotion(
        text
    )

    return jsonify({

        "sentiment":
        prediction.capitalize(),

        "confidence":
        confidence,

        "emotion":
        emotion,

        "word_count":
        word_count,

        "reading_time":
        reading_time,

        "keywords":
        keywords,

        "sentence_analysis":
        sentence_results

    })


if __name__ == "__main__":

    app.run(
        debug=True
    )