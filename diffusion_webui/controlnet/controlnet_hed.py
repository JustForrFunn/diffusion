from diffusers import ( StableDiffusionControlNetPipeline, 
                       ControlNetModel, UniPCMultistepScheduler)

from controlnet_aux import HEDdetector
from PIL import Image
import gradio as gr
import torch

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base"
]

stable_prompt_list = [
        "a photo of a man.",
        "a photo of a girl."
    ]

stable_negative_prompt_list = [
        "bad, ugly",
        "deformed"
    ]


def controlnet_hed(image_path:str):
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    image = Image.open(image_path)
    image = hed(image)

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-hed", 
        torch_dtype=torch.float16
    )
    return controlnet, image


def stable_diffusion_controlnet_hed(
    image_path:str,
    model_path:str,
    prompt:str,
    negative_prompt:str,
    guidance_scale:int,
    num_inference_step:int,
    ):

    controlnet, image = controlnet_hed(image_path=image_path)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path, 
        controlnet=controlnet, 
        safety_checker=None, 
        torch_dtype=torch.float16
    )

    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    output = pipe(
        prompt = prompt,
        image = image,
        negative_prompt = negative_prompt,
        num_inference_steps = num_inference_step,
        guidance_scale = guidance_scale,
    ).images

    return output[0]

def stable_diffusion_controlnet_hed_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                controlnet_hed_image_file = gr.Image(
                    type='filepath', 
                    label='Image'
                )

                controlnet_hed_model_id = gr.Dropdown(
                    choices=stable_model_list, 
                    value=stable_model_list[0], 
                    label='Stable Model Id'
                )

                controlnet_hed_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_prompt_list[0], 
                    label='Prompt'
                )

                controlnet_hed_negative_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_negative_prompt_list[0], 
                    label='Negative Prompt'
                )

                with gr.Accordion("Advanced Options", open=False):
                    controlnet_hed_guidance_scale = gr.Slider(
                        minimum=0.1, 
                        maximum=15, 
                        step=0.1, 
                        value=7.5, 
                        label='Guidance Scale'
                    )

                    controlnet_hed_num_inference_step = gr.Slider(
                        minimum=1, 
                        maximum=100, 
                        step=1, 
                        value=50, 
                        label='Num Inference Step'
                    )

                controlnet_hed_predict = gr.Button(value='Generator')
            
        
            with gr.Column():
                output_image = gr.Image(label='Output')
        
        controlnet_hed_predict.click(
            fn=stable_diffusion_controlnet_hed,
            inputs=[
                controlnet_hed_image_file,
                controlnet_hed_model_id,
                controlnet_hed_prompt,
                controlnet_hed_negative_prompt,
                controlnet_hed_guidance_scale,
                controlnet_hed_num_inference_step,
            ],
            outputs=[output_image]
        )
        