import gradio as gr
import numpy as np
import requests
import os

API_URL = "http://3.144.149.104:8000//predict"

def call_api(image):
    image_path = "temp.jpg"
    image.save(image_path)

    with open(image_path, 'rb') as f:
        response = requests.post(
            API_URL,
            files= {"file": f}
        )
    return response.json()['predicted_class_scratch'], response.json()["Example Dinosaurs_scratch"], response.json()["Fun Fact!_scratch"], response.json()['predicted_class_finetuned'], response.json()["Example Dinosaurs_finetuned"], response.json()["Fun Fact!_finetuned"]

file_path = os.path.abspath("T-Rex-background.jpg")

custom_css = f"""
    .gradio-container {{
        background: url('/gradio_api/file={file_path}');
        background-size: cover;
    }}

"""

with gr.Blocks() as demo:
    gr.HTML(value = "<h1 style =  'font-size: 36px; text-align: center;'>Dinosaur Fossil Classifier</h1>")
    img = gr.Image(type = 'pil', width=500, height=500)

    evaluate_btn = gr.Button("Predict")
    with gr.Row():
        with gr.Column():
            gr.HTML(value = "<h1 style =  'font-size: 25px; text-align: center;'>Built from Scratch</h1>")
            predicted_result_scratch = gr.Textbox(label = "Fossil Type")
            example_scratch = gr.Textbox(label = "Dinosaur Examples")
            fun_fact_scratch = gr.Textbox(label = "Fun Fact!")
        with gr.Column():
            gr.HTML(value = "<h1 style =  'font-size: 25px; text-align: center;'>Fine-Tuned Pretrained Model</h1>")
            predicted_result_finetuned = gr.Textbox(label = "Fossil Type")
            example_finetuned = gr.Textbox(label = "Dinosaur Examples")
            fun_fact_finetuned = gr.Textbox(label = "Fun Fact!")

    evaluate_btn.click(fn = call_api, inputs = img, outputs = [predicted_result_scratch, example_scratch, fun_fact_scratch, predicted_result_finetuned, example_finetuned, fun_fact_finetuned])


demo.launch(theme=gr.themes.Ocean(), allowed_paths= [os.path.dirname(file_path)], server_name="0.0.0.0", server_port= 8000, css = custom_css)