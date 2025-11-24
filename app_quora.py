
import gradio as gr
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = r"C:\Users\sahil\Desktop\quora"

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "cv.pkl"), "rb") as f:
    cv = pickle.load(f)

def predict(q1, q2):
    q1_vec = cv.transform([q1])
    q2_vec = cv.transform([q2])
    score = cosine_similarity(q1_vec, q2_vec)[0][0]
    score = round(score, 3)

    if score >= 0.75:
        label = "✅ DUPLICATE (Same Meaning)"
    elif score >= 0.50:
        label = "⚠️ PARTIALLY SIMILAR"
    else:
        label = "❌ NOT DUPLICATE (Different Meaning)"

    return f"{label}\nSimilarity Score: {score}"

ui = gr.Interface(
    fn=predict,
    inputs=["text", "text"],
    outputs="text",
    title="Duplicate Question Checker",
    description="Enter two questions to check if they mean the same thing."
)

ui.launch()

