import gradio as gr
import numpy as np
import requests
import os

API_URL = "http://3.17.73.170:8000//predict"

def call_api(image):
    image_path = "temp.jpg"
    image.save(image_path)

    with open(image_path, 'rb') as f:
        response = requests.post(
            API_URL,
            files= {"file": f}
        )
    return response.json()['predicted_class'], response.json()["Example Dinosaurs"], response.json()["Fun Fact!"]

file_path = os.path.abspath("T-Rex-background.jpg")

custom_css = """
    .gradio-container {
        background-image: url('/gradio_api/file=app/T-Rex-background.jpg');
        background-size: cover;
    }

"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(value = "<h1 style =  'font-size: 36px; text-align: center;'>Dinosaur Fossil Classifier</h1>")

    img = gr.Image(type = 'pil', width=500, height=500)
    evaluate_btn = gr.Button("Predict")
    predicted_result = gr.Textbox(label = "Fossil Type")
    example = gr.Textbox(label = "Dinosaur Examples")
    fun_fact = gr.Textbox(label = "Fun Fact!")
    evaluate_btn.click(fn = call_api, inputs = img, outputs = [predicted_result, example, fun_fact])

demo.launch(theme=gr.themes.Ocean(), allowed_paths= [os.path.dirname(file_path)], server_name="0.0.0.0", server_port= 8000)



