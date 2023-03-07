from diffusers import StableDiffusionPipeline, DDIMScheduler
import gradio as gr
import torch

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base"
]

stable_inpiant_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting"
]

stable_prompt_list = [
        "a photo of a man.",
        "a photo of a girl."
    ]

stable_negative_prompt_list = [
        "bad, ugly",
        "deformed"
    ]

def stable_diffusion_text2img(
    model_path:str,
    prompt:str,
    negative_prompt:str,
    guidance_scale:int,
    num_inference_step:int,
    height:int,
    width:int,
    ):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        safety_checker=None, 
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
    ).images

    return images[0]

def stable_diffusion_text2img_app():
    with gr.Tab('Text2Image'):
        text2image_model_path = gr.Dropdown(
            choices=stable_model_list, 
            value=stable_model_list[0], 
            label='Text-Image Model Id'
        )

        text2image_prompt = gr.Textbox(
            lines=1, 
            value=stable_prompt_list[0], 
            label='Prompt'
        )

        text2image_negative_prompt = gr.Textbox(
            lines=1, 
            value=stable_negative_prompt_list[0], 
            label='Negative Prompt'
        )

        with gr.Accordion("Advanced Options", open=False):
            text2image_guidance_scale = gr.Slider(
                minimum=0.1, 
                maximum=15, 
                step=0.1, 
                value=7.5, 
                label='Guidance Scale'
            )

            text2image_num_inference_step = gr.Slider(
                minimum=1, 
                maximum=100, 
                step=1, 
                value=50, 
                label='Num Inference Step'
            )

            text2image_height = gr.Slider(
                minimum=128, 
                maximum=1280, 
                step=32, 
                value=512, 
                label='Image Height'
            )

            text2image_width = gr.Slider(
                minimum=128, 
                maximum=1280, 
                step=32, 
                value=768, 
                label='Image Height'
            )

        text2image_predict = gr.Button(value='Generator')
    
    variables = {
        "model_path": text2image_model_path,
        "prompt": text2image_prompt,
        "negative_prompt": text2image_negative_prompt,
        "guidance_scale": text2image_guidance_scale,
        "num_inference_step": text2image_num_inference_step,
        "height": text2image_height,
        "width": text2image_width,
        "predict": text2image_predict
    }

    return variables 
