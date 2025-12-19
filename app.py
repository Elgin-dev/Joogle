from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# --- Load trained model ---
with open("model_data.pkl", "rb") as f:
    trigrams = pickle.load(f)

# --- Dynamic learning buffer ---
session_buffer = {}

# --- Expanded sensitive word list ---
SENSITIVE_WORDS = [
    "hate", "kill", "bomb", "abuse", "terror", "racist", "murder", "drugs",
    "sex", "nude", "gun", "weapon", "fight", "suicide", "porn", "assault",
    "attack", "death", "violence", "blood", "explosive"
]

def is_sensitive(text):
    return any(word.lower() in text.lower() for word in SENSITIVE_WORDS)

def predict_next_word(input_text):
    words = input_text.lower().split()
    suggestions = []

    # 1️⃣ — model-based trigrams
    if len(words) >= 2:
        key = (words[-2], words[-1])
        if key in trigrams:
            suggestions.extend(trigrams[key])

    # 2️⃣ — learned buffer suggestions
    for key, vals in session_buffer.items():
        if input_text.lower().strip().startswith(key):
            for v in vals:
                if v not in suggestions:
                    suggestions.insert(0, v)  # learned = top priority

    # 3️⃣ — if still few suggestions, generate smart defaults
    if len(suggestions) < 5:
        first_word = words[0] if words else ""
        default_suggestions = []

        if first_word == "what":
            default_suggestions = [
                "what is the role of",
                "what is the country name of",
                "what are the advantages of",
                "what about karunya",
                "what is artificial intelligence"
            ]
        elif first_word == "how":
            default_suggestions = [
                "how to make a project in python",
                "how to cook noodles",
                "how to train a deep learning model",
                "how to go to karunya university",
                "how does AI work"
            ]
        elif first_word == "why":
            default_suggestions = [
                "why is the sky blue",
                "why do we use AI",
                "why python is popular",
                "why should we recycle",
                "why karunya is famous"
            ]
        elif first_word == "where":
            default_suggestions = [
                "where is karunya located",
                "where can I find machine learning datasets",
                "where is India on the map",
                "where do penguins live",
                "where is the nearest restaurant"
            ]
        elif first_word == "who":
            default_suggestions = [
                "who is the founder of karunya university",
                "who invented artificial intelligence",
                "who wrote the Bible",
                "who discovered gravity",
                "who is the prime minister of India"
            ]
        else:
            default_suggestions = [
                f"{input_text} meaning",
                f"{input_text} definition",
                f"{input_text} example",
                f"{input_text} explanation",
                f"{input_text} information"
            ]

        for s in default_suggestions:
            if s not in suggestions:
                suggestions.append(s)

    # 4️⃣ — remove sensitive/duplicates
    clean = []
    for s in suggestions:
        if not is_sensitive(s) and s not in clean:
            clean.append(s)

    return clean[:5]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/suggest")
def suggest():
    query = request.args.get("q", "")
    if not query:
        return jsonify([])
    suggestions = predict_next_word(query)
    return jsonify(suggestions)

@app.route("/teach", methods=["POST"])
def teach():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"message": "Empty text"})

    if is_sensitive(text):
        return jsonify({"message": "⚠️ Sensitive content detected! Not saved."})

    parts = text.split(" ", 1)
    if len(parts) > 1:
        key = parts[0].lower()
        session_buffer.setdefault(key, []).append(text)
        return jsonify({"message": f"✅ Learned: {text}"})
    return jsonify({"message": "Text too short to learn."})

@app.route("/clear", methods=["POST"])
def clear():
    session_buffer.clear()
    return jsonify({"message": "🧹 Cache memory cleared!"})

if __name__ == "__main__":
    app.run(debug=True)
