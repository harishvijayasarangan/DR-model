import onnxruntime
import numpy as np
import gradio as gr
from PIL import Image

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
session = onnxruntime.InferenceSession('model.onnx')

def transform_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.5353, 0.3628, 0.2486], dtype=np.float32)
    std = np.array([0.2126, 0.1586, 0.1401], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32) 
def predict(input_img):
    """Predict DR grade from input image using ONNX model"""
    input_tensor = transform_image(input_img)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: input_tensor})[0][0]
    exp_preds = np.exp(prediction - np.max(prediction))
    probabilities = exp_preds / exp_preds.sum()    
    confidences = {labels[i]: float(probabilities[i]) for i in labels}
    #filtered_confidences = {key: confidences[key] for key in ["No DR", "Severe"]}
    #return filtered_confidences
    return confidences

dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Detection",
    description="",
    examples=[
        "sample/1.jpeg",
        "sample/2.jpeg",
        "sample/3.jpeg",
        "sample/4.jpeg",
    ],
    analytics_enabled=False,
)
if __name__ == "__main__":
    dr_app.launch(server_name="0.0.0.0", server_port=8080)
