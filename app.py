from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load Saved Model and Vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    text_vector = tfidf.transform([text])

    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        result = "😊 Positive"

    elif prediction[0] == 0:
        result = "😐 Neutral"

    else:
        result = "😡 Negative"

    return render_template("index.html", prediction=result)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)