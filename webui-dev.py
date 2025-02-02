import tempfile
import traceback
import gradio as gr
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import ollama
import json
from functools import partial
import mlx.core as mx
import gc
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import re
import subprocess
from mflux.config.config import Config, ConfigControlnet, CustomModelConfig
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from tqdm import tqdm
from PIL import Image
from mflux.ui.cli.parsers import CommandLineParser
import base64
from io import BytesIO
import numpy as np

LORAS_DIR = os.path.join(os.path.dirname(__file__), "lora")
os.makedirs(LORAS_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)       
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# We will handle up to 5 LoRAs for scale sliders
MAX_LORAS = 5

def update_lora_scales(selected_loras):
    # We will return updates for up to 5 sliders
    updates = []
    for i, lora_name in enumerate(selected_loras[:MAX_LORAS]):
        updates.append(gr.update(visible=True, label=f"Scale: {lora_name}", value=1.0))
    # For any remaining sliders (not used), hide them
    for _ in range(MAX_LORAS - len(selected_loras)):
        updates.append(gr.update(visible=False, value=1.0, label="Scale:"))
    return updates

def get_lora_choices():
    return [name for name, _ in get_available_lora_files()]

def create_flux(model, type, path, quantize, lora_path_list, lora_scale_list):
    lora_paths = list(lora_path) if lora_paths_list else None
    lora_scales = list(lora_scales_list) if lora_scales_list else None
           
    model_config = get_custom_model_config()



    try:
        custom_config = get_custom_model_config(base_model)
        if base_model in ["dev", "schnell", "dev-8-bit", "dev-4-bit", "schnell-8-bit", "schnell-4-bit"]:
            model_path = None
        else:
            model_path = os.path.join(MODELS_DIR, base_model)
    except ValueError:
        custom_config = CustomModelConfig(base_model, base_model, 1000, 512)
        model_path = os.path.join(MODELS_DIR, base_model)
    
    if "-8-bit" in model:
        quantize = 8
    elif "-4-bit" in model:
        quantize = 4
    
    flux = FluxClass(
        model_config=custom_config,
        quantize=quantize,
        local_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    
    return flux

def get_available_lora_files():
    lora_files = []
    for root, dirs, files in os.walk(LORA_DIR):
        for file in files:
            if file.endswith(".safetensors"):
                display_name = os.path.splitext(file)[0]
                lora_files.append((display_name, os.path.join(root, file)))
    lora_files.sort(key=lambda x: x[0].lower())
    return lora_files

def get_available_models():
    standard_models = ["schnell-4-bit", "schnell-8-bit", "dev-4-bit", "dev-8-bit", "schnell", "dev"]
    custom_models = [f.name for f in Path(MODELS_DIR).iterdir() if f.is_dir()]
    return standard_models + custom_models

def ensure_ollama_model(model_name):
    try:
        ollama.pull(model_name)
        return True
    except Exception:
        return False

def load_ollama_settings():
    try:
        with open('ollama_settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        _, default_model = get_available_ollama_models()
        settings = {'model': default_model}
    
    settings['system_prompt'] = read_system_prompt()
    return settings

def create_ollama_settings():
    settings = load_ollama_settings()
    available_models, _ = get_available_ollama_models()
    ollama_model = gr.Dropdown(
        choices=available_models,
        value=settings['model'],
        label="Ollama Model"
    )
    system_prompt = gr.Textbox(
        label="System Prompt", lines=10, value=settings['system_prompt']
    )
    save_button = gr.Button("Save Ollama Settings")
    return [ollama_model, system_prompt, save_button]

def save_settings(model, prompt):
    save_ollama_settings(model, prompt)
    gr.Info("Settings saved!")
    model_update = gr.update(choices=get_available_ollama_models(), value=model)
    return gr.update(open=False)

def enhance_prompt(prompt, ollama_model, system_prompt):
    print(f"prompt={prompt}, model={ollama_model}, system_prompt={system_prompt}")
    try:
        response = ollama.generate(
            model=ollama_model,
            prompt=f"Enhance this prompt for an image generation AI: {prompt}",
            system=system_prompt,
            options={"temperature": 0.7}
        )
        enhanced_prompt = response['response'].strip()
        gr.Info(f"Prompt successfully improved with model {ollama_model}.")
        return enhanced_prompt
    except Exception as e:
        gr.Error(f"Error while improving prompt: {str(e)}")
        return prompt

        
        
def print_memory_usage(label):
    try:
        active_memory = mx.metal.get_active_memory() / 1e6
        peak_memory = mx.metal.get_peak_memory() / 1e6
        print(f"{label} - Active memory: {active_memory:.2f} MB, Peak memory: {peak_memory:.2f} MB")
    except AttributeError:
        print(f"{label} - Unable to get memory usage information")

# def generate_image_batch(
#     flux,
#     prompt,
#     seed,
#     steps,
#     height,
#     width,
#     guidance,
#     num_images
#     ):
#     images = []
#     filenames = []
#     for i in range(num_images):
#         current_seed = seed if seed is not None else int(time.time()) + i
#         output_filename = f"generated_{int(time.time())}_{i}.png"
#         output_path = os.path.join(OUTPUT_DIR, output_filename)

#         image = flux.generate_image(
#             seed=current_seed,
#             prompt=prompt,
#             config=Config(
#                 num_inference_steps=steps,
#                 height=height,
#                 width=width,
#                 guidance=guidance,
#             ),
#         )
#         image.image.save(output_path)
#         images.append(image)
#         filenames.append(output_filename)
#     return images, filenames

def generate_image_advanced_gradio(
    prompt,
    model,
    seed,
    height,
    width,
    steps,
    guidance,
    metadata,
    ollama_model,
    system_prompt, 
    num_images,
    lora_files,
    *lora_scales_list
):
    print(f"\n--- Generating image (Advanced) ---")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Adjusted Dimensions: {height}x{width}")
    print(f"Steps: {steps}")
    print(f"Guidance: {guidance}")
    print(f"LoRA files: {lora_files}")
    print(f"LoRA scales: {lora_scales_list}")
    print(f"Number of Images: {num_images}")
    print_memory_usage("Before generation")

    start_time = time.time()

    try:
        valid_loras = process_lora_files(lora_files)
        # match number of scales to the number of loras selected
        lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

        seed = None if seed == "" else int(seed)

        if not steps or steps.strip() == "":
            base_model = model.replace("-4-bit", "").replace("-8-bit", "")
            if "schnell" in base_model:
                steps = 4
            elif "dev" in base_model:
                steps = 20
            else:
                steps = 20
        else:
            steps = int(steps)

        flux = get_or_create_flux(model, None, None, valid_loras, lora_scales)

        print_memory_usage("After creating flux")

        images = []
        filenames = []
        for i in range(num_images):
            current_seed = int(time.time()) + i
            output_filename = f"generated_advanced_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                ),
            )

            image.image.save(output_path)
            images.append(image.image)
            filenames.append(output_filename)

        del flux
        gc.collect()
        force_mlx_cleanup()

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        # Return gallery of images, filenames, and prompt
        return images, "\n".join(filenames), prompt

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        traceback.print_exc()
        return [], "", prompt

    finally:
        force_mlx_cleanup()
        gc.collect()

def generate_image_controlnet_gradio(
    prompt,
    control_image,
    model,
    seed,
    height,
    width,
    steps,
    guidance,
    controlnet_strength,
    metadata,
    save_canny,
    ollama_model,
    system_prompt,
    num_images,
    lora_files,
    *lora_scales_list
):
    print(f"\n--- Generating image (ControlNet) ---")
    print(f"Received parameters:")
    print(f"- prompt: {prompt}")
    print(f"- model: {model}")
    print(f"- seed: {seed}")
    print(f"- height: {height}")
    print(f"- width: {width}") 
    print(f"- steps: {steps}")
    print(f"- guidance: {guidance}")
    print(f"- controlnet_strength: {controlnet_strength}")
    print(f"- lora_files: {lora_files}")
    print(f"- lora_scales_list: {lora_scales_list}")
    print(f"- save_canny: {save_canny}")
    print(f"- num_images: {num_images}")

    print_memory_usage("Before generation")
    start_time = time.time()
    generated_images = []
    filenames = []
    canny_image_to_return = None
    try:
        valid_loras = process_lora_files(lora_files)
        lora_scales = lora_scales_list[:len(valid_loras)] if valid_loras else None

        seed = None if seed == "" else int(seed)
        steps = None if steps == "" else int(steps)
        if steps is None:
            steps = 4 if "schnell" in model else 14

        flux = get_or_create_flux(
            model, 
            None, 
            None,
            valid_loras,
            lora_scales,
            is_controlnet=True
        )

        timestamp = int(time.time())
        control_image_path = os.path.join(OUTPUT_DIR, f"control_image_{timestamp}.png")
        control_image.save(control_image_path)

        # Generate multiple images
        for i in range(num_images):
            current_seed = seed if seed is not None else int(time.time()) + i
            output_filename = f"generated_controlnet_{int(time.time())}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            generated_image = flux.generate_image(
                seed=current_seed,
                prompt=prompt,
                controlnet_image_path=control_image_path,
                config=ConfigControlnet(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    controlnet_strength=controlnet_strength,
                ),
                output=output_path,
                controlnet_save_canny=save_canny
            )

            generated_images.append(generated_image.image)
            filenames.append(output_filename)

            # If canny is requested
            if save_canny:
                canny_path = output_path.replace('.png', '_controlnet_canny.png')
                if os.path.exists(canny_path):
                    canny_image_to_return = Image.open(canny_path)

        print_memory_usage("After generating images")

        # Cleanup
        if os.path.exists(control_image_path):
            os.remove(control_image_path)

        print(f"Generation completed in {time.time() - start_time:.2f}s")
        return generated_images, "\n".join(filenames), prompt, canny_image_to_return

    except Exception as e:
        print(f"\nError in ControlNet generation: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        return [], "", prompt, None

    finally:
        if 'flux' in locals():
            del flux
        gc.collect()
        force_mlx_cleanup()

def process_lora_files(selected_loras):
    if not selected_loras:
        return []
    lora_files = get_available_lora_files()
    if not lora_files:
        return []
    lora_dict = dict(lora_files)
    valid_loras = []
    for lora in selected_loras:
        if lora in lora_dict:
            valid_loras.append(lora_dict[lora])
    return valid_loras

def force_mlx_cleanup():
    mx.eval(mx.zeros(1))
    
    if hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()
    
    if hasattr(mx.metal, 'reset_peak_memory'):
        mx.metal.reset_peak_memory()

    gc.collect()

def get_updated_lora_files():
    lora_files = []
    for root, dirs, files in os.walk(LORA_DIR):
        for file in files:
            if file.endswith(".safetensors") or file.endswith(".ckpt"):
                lora_files.append(file)
    return lora_files

def get_updated_models():
    predefined_models = ["schnell-4-bit", "dev-4-bit", "schnell-8-bit", "dev-8-bit", "schnell", "dev"]
    custom_models = [f.name for f in Path(MODELS_DIR).iterdir() if f.is_dir()]
    custom_models = [m for m in custom_models if m not in predefined_models]
    custom_models.sort(key=str.lower)
    all_models = predefined_models + custom_models
    return all_models

def refresh_lora_choices():
    return gr.update(choices=[name for name, _ in get_available_lora_files()])

def update_dimensions_on_image_change(image):
    if image is not None:
        width, height = image.size
        return (
            gr.update(value=width),
            gr.update(value=height),
            width,
            height,
            gr.update(value=1.0),
        )
    else:
        return (
            gr.update(value=None),
            gr.update(value=None),
            None,
            None,
            gr.update(value=1.0),
        )

def create_ui():
    with gr.Blocks(css="""
        .refresh-button {
            background-color: white !important;
            border: 1px solid #ccc !important;
            color: black !important;
            padding: 0px 8px !important;
            height: 38px !important;
            margin-left: -10px !important;
        }
        .refresh-button:hover {
            background-color: #f0f0f0 !important;
        }
    """) as demo:
        with gr.Tabs():

            # MFLUX Easy Generate
            with gr.TabItem("MFLUX Easy Generate", id=0):
                with gr.Row():
                    with gr.Column():
                        prompt_easy = gr.Textbox(label="Prompt", lines=2)
                        with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_easy:
                            ollama_components_easy = create_ollama_settings()
                        with gr.Row():
                            enhance_ollama_easy = gr.Button("Enhance prompt with Ollama")
                        ollama_components_easy[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_easy[0], ollama_components_easy[1]],
                            outputs=[ollama_section_easy]
                        )
                        model_easy = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell-4-bit",
                            allow_custom_value=True
                        )
                        image_format_easy = gr.Dropdown(
                            choices=[
                                "Portrait (576x1024)",
                                "Landscape (1024x576)",
                                "Background (1920x1080)",
                                "Square (1024x1024)",
                                "Poster (1080x1920)",
                                "Wide Screen (2560x1440)",
                                "Ultra Wide Screen (3440x1440)",
                                "Banner (728x90)"
                            ],
                            label="Image Format",
                            value="Portrait (576x1024)"
                        )
                        with gr.Row():
                            lora_files_easy = gr.Dropdown(
                                choices=get_lora_choices(),
                                label="Select LoRA Files",
                                multiselect=True,
                                allow_custom_value=True,
                                value=[],
                                interactive=True,
                                scale=9
                            )
                            refresh_lora_easy = gr.Button(
                                "🔄",
                                variant='tool',
                                size='sm',
                                scale=1,
                                min_width=30,
                                elem_classes='refresh-button'
                            )
                        refresh_lora_easy.click(
                            fn=refresh_lora_choices,
                            inputs=[],
                            outputs=[lora_files_easy]
                        )
                        # Create LoRA scale sliders for up to 5 LoRAs
                        lora_scales_easy = [gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) for _ in range(MAX_LORAS)]
                        lora_files_easy.change(
                            fn=update_lora_scales,
                            inputs=[lora_files_easy],
                            outputs=lora_scales_easy
                        )
                        num_images_easy = gr.Number(label="Number of Images", value=1, precision=0)
                        generate_button_easy = gr.Button("Generate Image", variant='primary')
                        #generate_button_easy = gr.Button("Generate Image", variant='primary')
                    with gr.Column():
                        output_gallery_easy = gr.Gallery(label="Generated Images", )
                        output_filename_easy = gr.Textbox(label="Saved Image Filenames")
                enhance_ollama_easy.click(
                    fn=enhance_prompt,
                    inputs=[prompt_easy, ollama_components_easy[0], ollama_components_easy[1]],
                    outputs=prompt_easy
                )
                generate_button_easy.click(
                    fn=generate_image_easy_gradio,
                    inputs=[
                        prompt_easy,
                        model_easy,
                        image_format_easy,
                        ollama_components_easy[0],
                        ollama_components_easy[1], 
                        num_images_easy,
                        lora_files_easy, 
                        *lora_scales_easy
                    ],
                    outputs=[output_gallery_easy, output_filename_easy, prompt_easy]
                )

            # MFLUX Advanced Generate
            with gr.TabItem("MFLUX Advanced Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt_adv = gr.Textbox(label="Prompt", lines=2)
                        with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_adv:
                            ollama_components_adv = create_ollama_settings()
                            
                        with gr.Row():
                            enhance_ollama_adv = gr.Button("Enhance prompt with Ollama")
                        ollama_components_adv[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_adv[0], ollama_components_adv[1]],
                            outputs=[ollama_section_adv]
                        )
                        model_adv = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell-4-bit"
                        )
                        seed_adv = gr.Textbox(label="Seed (optional)", value="")
                        with gr.Row():
                            width_adv = gr.Number(label="Width", value=576, precision=0)
                            height_adv = gr.Number(label="Height", value=1024, precision=0)
                            
                        steps_adv = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance_adv = gr.Number(label="Guidance Scale", value=3.5, visible=False)
                        with gr.Row():
                            lora_files_adv = gr.Dropdown(
                                choices=get_lora_choices(),
                                label="Select LoRA Files",
                                multiselect=True,
                                allow_custom_value=True,
                                value=[],
                                interactive=True,
                                scale=9
                            )
                            refresh_lora_adv = gr.Button(
                                "🔄",
                                variant='tool',
                                size='sm',
                                scale=1,
                                min_width=30,
                                elem_classes='refresh-button'
                            )
                        refresh_lora_adv.click(
                            fn=refresh_lora_choices,
                            inputs=[],
                            outputs=[lora_files_adv]
                        )
                        # Lora scale sliders for advanced tab
                        lora_scales_adv = [gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) for _ in range(MAX_LORAS)]
                        lora_files_adv.change(
                            fn=update_lora_scales,
                            inputs=[lora_files_adv],
                            outputs=lora_scales_adv
                        )
                        num_images_adv = gr.Number(label="Number of Images", value=1, precision=0)
                        metadata_adv = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button_adv = gr.Button("Generate Image", variant='primary')
                        
                        with gr.Column():
                            output_gallery_adv = gr.Gallery(label="Generated Images")
                            output_filename_adv = gr.Textbox(label="Saved Image Filenames")
                model_adv.change(
                    fn=update_guidance_visibility,
                    inputs=[model_adv],
                    outputs=[guidance_adv]
                )
                enhance_ollama_adv.click(
                    fn=enhance_prompt,
                    inputs=[prompt_adv, ollama_components_adv[0], ollama_components_adv[1]],
                    outputs=prompt_adv
                )
                generate_button_adv.click(
                    fn=generate_image_advanced_gradio,
                    inputs=[
                        prompt_adv,
                        model_adv,
                        seed_adv,
                        width_adv,
                        height_adv,
                        steps_adv,
                        guidance_adv,
                        metadata_adv,
                        ollama_components_adv[0],
                        ollama_components_adv[1], 
                        num_images_adv,
                        lora_files_adv,
                        *lora_scales_adv
                    ],
                    outputs=[output_gallery_adv, output_filename_adv, prompt_adv]
                )

            # ControlNet
            with gr.TabItem("ControlNet"):
                with gr.Row():
                    with gr.Column():
                        prompt_cn = gr.Textbox(label="Prompt", lines=2)
                        with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_cn:
                            ollama_components_cn = create_ollama_settings()
                        with gr.Row():
                            enhance_ollama_cn = gr.Button("Enhance prompt with Ollama")
                        ollama_components_cn[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_cn[0], ollama_components_cn[1]],
                            outputs=[ollama_section_cn]
                        )
                        control_image_cn = gr.Image(label="Control Image", type="pil")
                        canny_image_cn = gr.Image(label="Canny Image", type="pil", visible=False)
                        width_cn = gr.Number(label="Width")
                        height_cn = gr.Number(label="Height")
                        scale_factor_cn = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Scale Factor (%)"
                        )
                        original_width_cn = gr.State()
                        original_height_cn = gr.State()
                        control_image_cn.change(
                            fn=update_dimensions_on_image_change,
                            inputs=[control_image_cn],
                            outputs=[width_cn, height_cn, original_width_cn, original_height_cn, scale_factor_cn]
                        )
                        scale_factor_cn.change(
                            fn=update_dimensions_on_scale_change,
                            inputs=[scale_factor_cn, original_width_cn, original_height_cn],
                            outputs=[width_cn, height_cn]
                        )
                        model_cn = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell-4-bit"
                        )
                        seed_cn = gr.Textbox(label="Seed (optional)", value="")
                        steps_cn = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance_cn = gr.Number(label="Guidance Scale", value=3.5, visible=False)
                        controlnet_strength_cn = gr.Number(label="ControlNet Strength", value=0.5)
                        with gr.Row():
                            lora_files_cn = gr.Dropdown(
                                choices=get_lora_choices(),
                                label="Select LoRA Files",
                                multiselect=True,
                                allow_custom_value=True,
                                value=[],
                                interactive=True,
                                scale=9
                            )
                            refresh_lora_cn = gr.Button(
                                "🔄",
                                variant='tool',
                                size='sm',
                                scale=1,
                                min_width=30,
                                elem_classes='refresh-button'
                            )
                        refresh_lora_cn.click(
                            fn=refresh_lora_choices,
                            inputs=[],
                            outputs=[lora_files_cn]
                        )

                        # Lora scale sliders for controlnet
                        lora_scales_cn = [gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) for _ in range(MAX_LORAS)]
                        lora_files_cn.change(
                            fn=update_lora_scales,
                            inputs=[lora_files_cn],
                            outputs=lora_scales_cn
                        )
                        num_images_cn = gr.Number(label="Number of Images", value=1, precision=0)
                        metadata_cn = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        save_canny_cn = gr.Checkbox(label="Save Canny Edge Detection Image", value=False)
                        generate_button_cn = gr.Button("Generate Image", variant='primary')
                        with gr.Column():
                            output_gallery_cn = gr.Gallery(label="Generated Images")
                            output_message_cn = gr.Textbox(label="Saved Image Filenames")
                            canny_image_cn = gr.Image(label="Canny Image", visible=False)
                        enhance_ollama_cn.click(
                            fn=enhance_prompt,
                            inputs=[prompt_cn, ollama_components_cn[0], ollama_components_cn[1]],
                            outputs=prompt_cn
                        )
                        generate_button_cn.click(
                            fn=generate_image_controlnet_gradio,
                            inputs=[
                                prompt_cn,
                                control_image_cn,
                                model_cn,
                                seed_cn,
                                height_cn,
                                width_cn,
                                steps_cn, 
                                guidance_cn,
                                controlnet_strength_cn,
                                metadata_cn,
                                save_canny_cn,
                                ollama_components_cn[0],
                                ollama_components_cn[1], 
                                num_images_cn,
                                lora_files_cn,
                                *lora_scales_cn
                            ],
                            outputs=[output_gallery_cn, output_message_cn, prompt_cn, canny_image_cn]
                        )
                        gr.Markdown("""
                        ⚠ Note: Controlnet requires [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. 
                        It can work well with `schnell`, but performance is not guaranteed.
        
                        ⚠ Note: The output can be highly sensitive to the controlnet strength and is very much dependent on the reference image. 
                        Too high settings will corrupt the image. A recommended starting point is a value like 0.4. Experiment with different strengths to find the best result.
                        """)
                        save_canny_cn.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=save_canny_cn,
                            outputs=canny_image_cn
                        )

            # Image-to-Image
            with gr.TabItem("Image-to-Image", id=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_i2i = gr.Textbox(label="Prompt", lines=2)
                        with gr.Accordion("⚙️ Ollama Settings", open=False) as ollama_section_i2i:
                            ollama_components_i2i = create_ollama_settings()
                        with gr.Row():
                            enhance_ollama_i2i = gr.Button("Enhance prompt with Ollama")
                        ollama_components_i2i[2].click(
                            fn=save_settings,
                            inputs=[ollama_components_i2i[0], ollama_components_i2i[1]],
                            outputs=[ollama_section_i2i]
                        )
                        init_image_i2i = gr.Image(label="Initial Image", type='pil')
                        init_image_strength_i2i = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.01,
                            label="Init Image Strength"
                        )
                        width_i2i = gr.Number(label="Width")
                        height_i2i = gr.Number(label="Height")
                        scale_factor_i2i = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Scale Factor (%)"
                        )
                        original_width_i2i = gr.State()
                        original_height_i2i = gr.State()
                        init_image_i2i.change(
                            fn=update_dimensions_on_image_change,
                            inputs=[init_image_i2i],
                            outputs=[width_i2i, height_i2i, original_width_i2i, original_height_i2i, scale_factor_i2i],
                        )
                        scale_factor_i2i.change(
                            fn=update_dimensions_on_scale_change,
                            inputs=[scale_factor_i2i, original_width_i2i, original_height_i2i],
                            outputs=[width_i2i, height_i2i]
                        )
                        model_i2i = gr.Dropdown(
                            choices=get_updated_models(),
                            label="Model",
                            value="schnell-4-bit"
                        )
                        seed_i2i = gr.Textbox(label="Seed (optional)", value="")
                        steps_i2i = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance_i2i = gr.Number(label="Guidance Scale", value=3.5, visible=False)
                        # lora scale sliders i2i
                        lora_files_i2i = gr.Dropdown(
                            choices=get_lora_choices(),
                            label="Select LoRA Files",
                            multiselect=True,
                            allow_custom_value=True,
                            value=[],
                            interactive=True,
                            scale=9
                        )
                        refresh_lora_i2i = gr.Button(
                            "🔄",
                            variant='tool',
                            size='sm',
                            scale=1,
                            min_width=30,
                            elem_classes='refresh-button'
                        )
                        refresh_lora_i2i.click(
                            fn=refresh_lora_choices,
                            inputs=[],
                            outputs=[lora_files_i2i]
                        )
                        lora_scales_i2i = [gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="Scale:", visible=False, value=1.0) for _ in range(MAX_LORAS)]
                        lora_files_i2i.change(
                            fn=update_lora_scales,
                            inputs=[lora_files_i2i],
                            outputs=lora_scales_i2i
                        )
                        num_images_i2i = gr.Number(label="Number of Images", value=1, precision=0)
                        metadata_i2i = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button_i2i = gr.Button("Generate Image", variant='primary')
                    with gr.Column(scale=1):
                        output_gallery_i2i = gr.Gallery(label="Generated Images")
                        output_filename_i2i = gr.Textbox(label="Saved Image Filenames")
                enhance_ollama_i2i.click(
                    fn=enhance_prompt,
                    inputs=[prompt_i2i, ollama_components_i2i[0], ollama_components_i2i[1]],
                    outputs=prompt_i2i
                )
                generate_button_i2i.click(
                    fn=generate_image_i2i_gradio,
                    inputs=[
                        prompt_i2i,
                        init_image_i2i,
                        init_image_strength_i2i,
                        model_i2i,
                        seed_i2i,
                        width_i2i,
                        height_i2i,
                        steps_i2i,
                        guidance_i2i,
                        metadata_i2i,
                        ollama_components_i2i[0],
                        ollama_components_i2i[1],
                        num_images_i2i,
                        lora_files_i2i,
                        *lora_scales_i2i
                    ],
                    outputs=[output_gallery_i2i, output_filename_i2i, prompt_i2i]
                )

            # Model & LoRA Management
            with gr.TabItem("Model & LoRA Management"):
                gr.Markdown("### Download LoRA")
                lora_source = gr.Radio(
                    choices=["CivitAI", "HuggingFace"],
                    label="LoRA Source",
                    value="CivitAI"
                )
                lora_input = gr.Textbox(label="LoRA Model Page URL (CivitAI) or Model Name (HuggingFace)")

                with gr.Accordion("API Key Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            api_key_input = gr.Textbox(
                                label="CivitAI API Key", 
                                type="password", 
                                value=load_api_key("civitai")
                            )
                            civitai_api_key_status = gr.Markdown(
                                value=f"CivitAI API Key Status: {'Saved' if load_api_key('civitai') else 'Not saved'}"
                            )
                            
                            hf_api_key_input = gr.Textbox(
                                label="HuggingFace API Key", 
                                type="password", 
                                value=load_api_key("huggingface")
                            )
                            hf_api_key_status = gr.Markdown(
                                value=f"HuggingFace API Key Status: {'Saved' if load_api_key('huggingface') else 'Not saved'}"
                            )
                        with gr.Column(scale=1):
                            save_api_keys_button = gr.Button("Save API Keys")
                            clear_api_keys_button = gr.Button("Clear API Keys")
                    gr.Markdown("Don't have an API key? [CivitAI](https://civitai.com/user/account), [HuggingFace](https://huggingface.co/settings/tokens)")

                download_lora_button = gr.Button("Download LoRA", variant='primary')
                lora_download_status = gr.Textbox(label="Download Status", lines=0.5)

                def save_api_keys(api_key_civitai, api_key_hf):
                    save_api_key(api_key_civitai, "civitai")
                    save_api_key(api_key_hf, "huggingface")
                    return (
                        f"CivitAI API Key Status: {'Saved successfully' if api_key_civitai else 'Not saved'}", 
                        f"HuggingFace API Key Status: {'Saved successfully' if api_key_hf else 'Not saved'}"
                    )

                def clear_api_keys():
                    save_api_key("", "civitai")
                    save_api_key("", "huggingface")
                    return "CivitAI API Key Status: Cleared", "HuggingFace API Key Status: Cleared"

                save_api_keys_button.click(
                    fn=save_api_keys,
                    inputs=[api_key_input, hf_api_key_input],
                    outputs=[civitai_api_key_status, hf_api_key_status]
                )

                clear_api_keys_button.click(
                    fn=clear_api_keys,
                    inputs=[],
                    outputs=[civitai_api_key_status, hf_api_key_status]
                )

                def download_lora(model_url_or_name, api_key_civitai, api_key_hf, lora_source):
                    if lora_source == "CivitAI":
                        return download_lora_model(model_url_or_name, api_key_civitai)
                    elif lora_source == "HuggingFace":
                        return download_lora_model_huggingface(model_url_or_name, api_key_hf)
                    else:
                        return gr.update(), gr.update(), gr.update(), "Error: Invalid LoRA source selected"

                download_lora_button.click(
                    fn=download_lora,
                    inputs=[lora_input, api_key_input, hf_api_key_input, lora_source],
                    outputs=[lora_files_easy, lora_files_adv, lora_files_cn, lora_files_i2i, lora_download_status]
                )

                gr.Markdown("## Download and Add Model")
                with gr.Row():
                    with gr.Column(scale=2):
                        hf_model_name = gr.Textbox(label="Hugging Face Model Name")
                        alias = gr.Textbox(label="Model Alias")
                        num_train_steps = gr.Number(label="Number of Training Steps", value=1000)
                        max_sequence_length = gr.Number(label="Max Sequence Length", value=512)
                        
                        with gr.Accordion("API Key Settings", open=False):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    hf_api_key_input = gr.Textbox(
                                        label="HuggingFace API Key", 
                                        type="password", 
                                        value=load_api_key("huggingface")
                                    )
                                    hf_api_key_status = gr.Markdown(
                                        value=f"HuggingFace API Key Status: {'Saved' if load_api_key('huggingface') else 'Not saved'}"
                                    )
                                with gr.Column(scale=1):
                                    save_hf_api_key_button = gr.Button("Save API Key")
                                    clear_hf_api_key_button = gr.Button("Clear API Key")
                            gr.Markdown("Don't have an API key? [Create one here](https://huggingface.co/settings/tokens)")
                        
                    with gr.Column(scale=1):
                        download_button = gr.Button("Download and Add Model", variant='primary')
                        download_output = gr.Textbox(label="Download Status", lines=3)

                def save_hf_api_key_handler(key):
                    save_api_key(key, "huggingface")
                    return f"HuggingFace API Key Status: {'Saved successfully' if key else 'Not saved'}"

                def clear_hf_api_key_handler():
                    save_api_key("", "huggingface")
                    return "HuggingFace API Key Status: Cleared"

                save_hf_api_key_button.click(
                    fn=save_hf_api_key_handler,
                    inputs=[hf_api_key_input],
                    outputs=[hf_api_key_status]
                )

                clear_hf_api_key_button.click(
                    fn=clear_hf_api_key_handler,
                    inputs=[],
                    outputs=[hf_api_key_status]
                )

                download_button.click(
                    fn=download_and_save_model,
                    inputs=[hf_model_name, alias, num_train_steps, max_sequence_length, hf_api_key_input],
                    outputs=[download_output]
                )

                gr.Markdown("## Quantize Model")
                with gr.Row():
                    with gr.Column(scale=2):
                        model_quant = gr.Dropdown(
                            choices=[m for m in get_updated_models() if not m.endswith("-4-bit") and not m.endswith("-8-bit")],
                            label="Model to Quantize",
                            value="dev"
                        )
                        quantize_level = gr.Radio(choices=["4", "8"], label="Quantize Level", value="8")
                    with gr.Column(scale=1):
                        save_button = gr.Button("Save Quantized Model", variant='primary')
                        save_output = gr.Textbox(label="Quantization Output", lines=3)

                save_button.click(
                    fn=save_quantized_model_gradio,
                    inputs=[model_quant, quantize_level],
                    outputs=[model_easy, model_adv, model_cn, model_i2i, model_quant, save_output]
                )

    return demo

demo = create_ui()

if __name__ == "__main__":
    demo.queue().launch(show_error=True)
