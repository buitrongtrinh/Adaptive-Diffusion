import time
import numpy as np
import random
from pathlib import Path
import torch
import gradio as gr
import gc
import sys

sys.path.append(str(Path(__file__).resolve().parent))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_memory():
    """Giải phóng VRAM để tránh OOM trên T4"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def generate_image(pipe, prompt, steps, is_adaptive=False):
    """Generate image với memory management"""
    try:
        start_time = time.time()
        image = pipe(prompt, num_inference_steps=steps, output_type="pil").images[0]
        gen_time = time.time() - start_time
        
        if is_adaptive and hasattr(pipe, 'reset_cache'):
            pipe.reset_cache()
        
        return image, gen_time, None
    except Exception as e:
        return None, 0, str(e)

def create_pipeline(model_name, threshold=None, max_skip_steps=None, is_adaptive=False):
    """Tạo một pipeline với tối ưu cho T4"""
    clear_memory()
    
    try:
        # Cấu hình tối ưu cho T4
        torch_dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name == 'stable_diffusion_v1-5':
            model_id = "runwayml/stable-diffusion-v1-5"
            if is_adaptive:
                from acceleration.sparse_pipeline import StableDiffusionPipeline as AdaptivePipeline
                pipe = AdaptivePipeline.from_pretrained(
                    model_id,
                    threshold=threshold,
                    max_skip_steps=max_skip_steps,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,  # Tối ưu RAM
                )
            else:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                )
        
        elif model_name == 'stable_diffusion_xl':
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            if is_adaptive:
                from acceleration.sparse_pipeline import StableDiffusionXLPipeline as AdaptivePipeline
                pipe = AdaptivePipeline.from_pretrained(
                    model_id,
                    threshold=threshold,
                    max_skip_steps=max_skip_steps,
                    torch_dtype=torch_dtype,
                    variant="fp16",
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                )
            else:
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    variant="fp16",
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                )
        else:
            return None, "Model not supported."
        
        pipe = pipe.to(device)
        
        # Tối ưu VRAM cho T4
        pipe.enable_attention_slicing(1)  # Giảm VRAM usage
        pipe.enable_vae_slicing()  # VAE slicing
        
        # Nếu có xformers, enable để tăng tốc
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass  # xformers không available
        
        return pipe, None
        
    except Exception as e:
        return None, f"Error creating pipeline: {str(e)}"

def set_scheduler(pipe, scheduler_name):
    """Thay đổi scheduler cho pipeline"""
    try:
        if scheduler_name == 'ddim':
            from diffusers import DDIMScheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif scheduler_name == 'dpm':
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler_name == 'euler':
            from diffusers import EulerDiscreteScheduler
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            return None, "Scheduler not supported."
        return pipe, None
    except Exception as e:
        return None, f"Error setting scheduler: {str(e)}"

def main():
    model_options = [
        "stable_diffusion_v1-5",
        "stable_diffusion_xl"
    ]

    scheduler_options = [
        "ddim",
        "dpm",
        "euler"
    ]
    
    # Default settings tối ưu cho T4
    default_model_name = 'stable_diffusion_xl'  # SD 1.5 nhẹ hơn SDXL
    default_scheduler_name = 'euler'  # DPM nhanh hơn
    default_threshold = 0.01
    default_max_skip_steps = 4
    
    with gr.Blocks(title="AdaptiveDiffusion Demo - T4 Optimized") as demo:
        gr.Markdown("""
        ## AdaptiveDiffusion vs Full-step Diffusion (T4 GPU Optimized)
        **Note**: Pipelines được load tuần tự để tránh OOM trên T4 (16GB VRAM)
                    
        Select model:
        -  `stable_diffusion_v1-5`: Nhẹ, nhanh, VRAM thấp, chất lượng vừa đủ
        -  `stable_diffusion_xl`: Chất lượng cao, chi tiết tốt, tốn VRAM & chậm hơn
                    
        Select scheduler:
        -  `ddim`: Nhanh, mượt, steps ít, chi tiết vừa
        -  `dpm`: Cân bằng tốc độ & chất lượng, tốt cho T4
        -  `euler`: Chất lượng cao, chi tiết tốt, chậm hơn, VRAM nhiều
                    
        """)
        
        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(
                    model_options, 
                    value=default_model_name, 
                    label="Select Model",
                    interactive=True,
                    show_label=True,
                )
                scheduler_select = gr.Dropdown(
                    scheduler_options, 
                    value=default_scheduler_name, 
                    label="Select Scheduler",
                    interactive=True,
                    show_label=True,
                )
            
            with gr.Column():
                seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                steps_input = gr.Slider(
                    minimum=10, 
                    maximum=50, 
                    value=50,  # Giảm default steps cho nhanh hơn
                    step=1,
                    label="Number of Steps"
                )
            
            with gr.Column():
                threshold_input = gr.Number(
                    value=default_threshold, 
                    label="Threshold", 
                    precision=3
                )
                max_skip_steps_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=default_max_skip_steps,
                    step=1,
                    label="Max Skip Steps"
                )
        
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="Enter your prompt here",
                value="Dogs playing poker in a saloon"
            )
        
        with gr.Row():
            generate_button = gr.Button("🚀 Generate Images", variant="primary")
        
        with gr.Row():
            error_output = gr.Textbox(label="Status/Errors", interactive=False)
        
        with gr.Row():
            memory_info = gr.Textbox(label="GPU Memory Info", interactive=False)
        
        with gr.Row():
            with gr.Column():
                original_image_output = gr.Image(label="Full-step Diffusion Output")
                original_time_output = gr.Textbox(
                    label="Full-step Time (s)", 
                    interactive=False
                )
            
            with gr.Column():
                adaptive_image_output = gr.Image(label="AdaptiveDiffusion Output")
                adaptive_time_output = gr.Textbox(
                    label="AdaptiveDiffusion Time (s)", 
                    interactive=False
                )
        
        with gr.Row():
            speedup_output = gr.Textbox(label="Speedup", interactive=False)
        
        # State management - chỉ lưu config, không lưu pipelines
        state = gr.State({
            'model_name': default_model_name,
            'scheduler_name': default_scheduler_name,
            'threshold': default_threshold,
            'max_skip_steps': default_max_skip_steps
        })
        
        def get_memory_info():
            """Lấy thông tin VRAM"""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"VRAM: {allocated:.2f}GB / {total:.2f}GB (Reserved: {reserved:.2f}GB)"
            return "CUDA not available"
        
        def update_interface(prompt, model_name, scheduler_name, seed, steps, 
                           threshold, max_skip_steps, state):
            """Generate images tuần tự để tránh OOM"""
            
            # Update state
            state['model_name'] = model_name
            state['scheduler_name'] = scheduler_name
            state['threshold'] = threshold
            state['max_skip_steps'] = max_skip_steps
            
            set_random_seed(int(seed))
            steps = int(steps)
            
            # === STEP 1: Generate Original Image ===
            yield (
                None, None,
                None, None,
                None, 
                "⏳ Loading Full-step diffusion pipeline...",
                get_memory_info(),
                state
            )
            
            pipe_original, error = create_pipeline(model_name, is_adaptive=False)
            if error:
                yield (None, None, None, None, None, error, get_memory_info(), state)
                return
            
            pipe_original, error = set_scheduler(pipe_original, scheduler_name)
            if error:
                yield (None, None, None, None, None, error, get_memory_info(), state)
                return
            
            yield (
                None, None,
                None, None,
                None,
                "🎨 Generating Full-step diffusion image...",
                get_memory_info(),
                state
            )
            
            original_image, original_time, error = generate_image(
                pipe_original, prompt, steps, is_adaptive=False
            )
            
            if error:
                yield (None, None, None, None, None, error, get_memory_info(), state)
                return
            
            # Xóa pipeline original để giải phóng VRAM
            del pipe_original
            clear_memory()
            
            yield (
                original_image, 
                f"{original_time:.2f} seconds",
                None, None,
                None,
                "⏳ Pipeline cleared, loading Adaptive Diffusion  pipeline...",
                get_memory_info(),
                state
            )
            
            # === STEP 2: Generate Adaptive Image ===
            pipe_adaptive, error = create_pipeline(
                model_name, threshold, max_skip_steps, is_adaptive=True
            )
            if error:
                yield (
                    original_image, f"{original_time:.2f} seconds",
                    None, None, None, error, get_memory_info(), state
                )
                return
            
            pipe_adaptive, error = set_scheduler(pipe_adaptive, scheduler_name)
            if error:
                yield (
                    original_image, f"{original_time:.2f} seconds",
                    None, None, None, error, get_memory_info(), state
                )
                return
            
            yield (
                original_image, 
                f"{original_time:.2f} seconds",
                None, None,
                None,
                "🚀 Generating Adaptive Diffusion image...",
                get_memory_info(),
                state
            )
            
            adaptive_image, adaptive_time, error = generate_image(
                pipe_adaptive, prompt, steps, is_adaptive=True
            )
            
            # Xóa pipeline adaptive
            del pipe_adaptive
            clear_memory()
            
            if error:
                yield (
                    original_image, f"{original_time:.2f} seconds",
                    None, None, None, error, get_memory_info(), state
                )
                return
            
            # Calculate speedup
            speedup = original_time / adaptive_time if adaptive_time > 0 else 0
            speedup_text = f"🚀 {speedup:.2f}x faster ({original_time - adaptive_time:.2f}s saved)"
            
            yield (
                original_image, 
                f"{original_time:.2f} seconds",
                adaptive_image, 
                f"{adaptive_time:.2f} seconds",
                speedup_text,
                "✅ Generation completed successfully!",
                get_memory_info(),
                state
            )
        
        generate_button.click(
            update_interface,
            inputs=[
                prompt_input, model_select, scheduler_select,
                seed_input, steps_input,
                threshold_input, max_skip_steps_input,
                state
            ],
            outputs=[
                original_image_output, original_time_output,
                adaptive_image_output, adaptive_time_output,
                speedup_output, error_output, memory_info, state
            ]
        )
        
        # Load memory info khi khởi động
        demo.load(lambda: get_memory_info(), None, memory_info)
    
    # Launch với settings cho Colab
    demo.launch(
        share=True,  # Tạo public link
        debug=True,
        server_name="0.0.0.0",  # Cho phép truy cập từ bên ngoài
        server_port=7860
    )

if __name__ == '__main__':
    main()