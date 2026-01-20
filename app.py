from flask import Flask, request, render_template_string
import pickle

# Load trained ML model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Tiny Mood Friend</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      background: linear-gradient(180deg, #fcefee, #e8f3ff);
      display: grid;
      place-items: center;
      height: 100vh;
      margin: 0;
    }
    .card {
      background: white;
      padding: 24px;
      border-radius: 20px;
      width: 320px;
      box-shadow: 0 20px 40px rgba(0,0,0,.15);
      text-align: center;
    }
    textarea {
      width: 100%;
      height: 100px;
      border-radius: 12px;
      border: 1px solid #ddd;
      padding: 10px;
      font-size: 14px;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border-radius: 999px;
      border: none;
      background: #ffcad4;
      font-weight: 600;
      cursor: pointer;
    }
    .result {
      margin-top: 16px;
      font-size: 18px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>🤍 Tiny Mood Friend</h2>
    <form method="post">
      <textarea name="text" placeholder="Write how you feel..."></textarea>
      <br />
      <button type="submit">Analyze Mood</button>
    </form>

    {% if mood %}
      <div class="result">
        <p>{{ emoji }} <strong>{{ mood }}</strong></p>
        <p>{{ advice }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

def predict_emotion(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

@app.route("/", methods=["GET", "POST"])
def index():
    mood = emoji = advice = None

    if request.method == "POST":
        text = request.form["text"]
        emotion = predict_emotion(text)

        if emotion == "happy":
            mood = "Happy"
            emoji = "😊"
            advice = "Enjoy the good moments!"
        elif emotion == "sad":
            mood = "Sad"
            emoji = "😢"
            advice = "Be gentle with yourself."
        elif emotion == "angry":
            mood = "Angry"
            emoji = "😡"
            advice = "Pause and breathe before reacting."
        elif emotion == "stressed":
            mood = "Stressed"
            emoji = "😰"
            advice = "Take a break. You’re doing your best."
        else:
            mood = "Calm"
            emoji = "😌"
            advice = "Everything feels peaceful."

    return render_template_string(
        HTML, mood=mood, emoji=emoji, advice=advice
    )

if __name__ == "__main__":
    app.run(debug=True)
