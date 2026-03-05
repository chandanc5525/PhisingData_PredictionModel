import joblib
import gradio as gr
import numpy as np

# Load trained model
model = joblib.load("../artifacts/model.pkl")

def predict(*features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Phishing Website" if prediction == 1 else "Legitimate Website"

# Create Gradio Interface
inputs = [gr.Number(label=f"Feature {i}") for i in range(1, 21)]  # adjust feature count
output = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="Phishing Website Detection",
    description="Enter website feature values to predict phishing status."
)

if __name__ == "__main__":
    app.launch()