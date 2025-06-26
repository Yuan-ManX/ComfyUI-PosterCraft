import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse
import datetime


class QwenRecapAgent:
    def __init__(self, model_path, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_kwargs = {"torch_dtype": "auto"}
        if device != "auto":
            model_kwargs["device_map"] = None
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if device != "auto":
            self.model.to(device)
        
        self.prompt_template = """You are an expert poster prompt designer. Your task is to rewrite a user's short poster prompt into a detailed and vivid long-format prompt. Follow these steps carefully:

    **Step 1: Analyze the Core Requirements**
    Identify the key elements in the user's prompt. Do not miss any details.
    - **Subject:** What is the main subject? (e.g., a person, an object, a scene)
    - **Style:** What is the visual style? (e.g., photorealistic, cartoon, vintage, minimalist)
    - **Text:** Is there any text, like a title or slogan?
    - **Color Palette:** Are there specific colors mentioned?
    - **Composition:** Are there any layout instructions?

    **Step 2: Expand and Add Detail**
    Elaborate on each core requirement to create a rich description.
    - **Do Not Omit:** You must include every piece of information from the original prompt.
    - **Enrich with Specifics:** Add professional and descriptive details.
        - **Example:** If the user says "a woman with a bow", you could describe her as "a young woman with a determined expression, holding a finely crafted wooden longbow, with an arrow nocked and ready to fire."
    - **Fill in the Gaps:** If the original prompt is simple (e.g., "a poster for a coffee shop"), use your creativity to add fitting details. You might add "The poster features a top-down view of a steaming latte with delicate art on its foam, placed on a rustic wooden table next to a few scattered coffee beans."

    **Step 3: Handle Text Precisely**
    - **Identify All Text Elements:** Carefully look for any text mentioned in the prompt. This includes:
        - **Explicit Text:** Subtitles, slogans, or any text in quotes.
        - **Implicit Titles:** The name of an event, movie, or product is often the main title. For example, if the prompt is "generate a 'Inception' poster ...", the title is "Inception".
    - **Rules for Text:**
        - **If Text Exists:**
            - You must use the exact text identified from the prompt.
            - Do NOT add new text or delete existing text.
            - Describe each text's appearance (font, style, color, position). Example: `The title 'Inception' is written in a bold, sans-serif font, integrated into the cityscape.`
        - **If No Text Exists:**
            - Do not add any text elements. The poster must be purely visual.
    - Most posters have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

    **Step 4: Final Output Rules**
    - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
    - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
    - **Keep it concise.** The final prompt should be under 300 words.

    ---
    **User Prompt:**
    {brief_description}"""
    
    def recap_prompt(self, original_prompt):
        full_prompt = self.prompt_template.format(brief_description=original_prompt)
        messages = [{"role": "user", "content": full_prompt}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096, temperature=0.6)
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            full_response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            final_answer = self._extract_final_answer(full_response)
            
            if final_answer:
                return final_answer.strip()
            
            print("Qwen returned an empty answer. Using original prompt.")
            return original_prompt
        except Exception as e:
            print(f"Qwen recap failed: {e}. Using original prompt.")
            return original_prompt

    def _extract_final_answer(self, full_response):
        if "</think>" in full_response:
            return full_response.split("</think>")[-1].strip()
        if "<think>" not in full_response:
            return full_response.strip()
        return None


class PosterGenerator:
    def __init__(self, 
                 pipeline_path="black-forest-labs/FLUX.1-dev",
                 custom_transformer_path=None,
                 qwen_model_path=None,
                 device="cuda:0"):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Load Qwen model for prompt rewriting
        if qwen_model_path and os.path.exists(qwen_model_path):
            print(f"Loading Qwen model from: {qwen_model_path}")
            self.qwen_agent = QwenRecapAgent(qwen_model_path, device=self.device)
        else:
            self.qwen_agent = None
        
        # Load Flux pipeline
        print(f"Loading Flux pipeline from: {pipeline_path}")
        self.pipeline = FluxPipeline.from_pretrained(pipeline_path, torch_dtype=torch.bfloat16)
        
        # Load custom transformer if provided
        if custom_transformer_path:
            print(f"Loading custom transformer from: {custom_transformer_path}")
            self.pipeline.transformer = FluxTransformer2DModel.from_pretrained(
                custom_transformer_path, 
                torch_dtype=torch.bfloat16
            )
        
        self.pipeline.to(self.device)
    
    def generate(self, 
                 prompt, 
                 enable_recap=True,
                 width=832, 
                 height=1216, 
                 num_inference_steps=28, 
                 guidance_scale=3.5, 
                 seed=None):
        
        # Prompt rewriting
        if enable_recap and self.qwen_agent:
            final_prompt = self.qwen_agent.recap_prompt(prompt)
        else:
            final_prompt = prompt
        
        # Poster generation
        if seed is None:
            seed = random.randint(1, 2**32 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.inference_mode():
            image = self.pipeline(
                prompt=final_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        
        return image, final_prompt, seed


def main():
    parser = argparse.ArgumentParser(description='Generate poster')
    parser.add_argument('--prompt', type=str, required=True, help='Input poster description prompt')
    parser.add_argument('--enable_recap', action='store_true', default=True, help='Enable prompt rewriting (default: True)')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--pipeline_path', type=str, default="black-forest-labs/FLUX.1-dev", help='Flux pipeline path')
    parser.add_argument('--custom_transformer_path', type=str, default="PosterCraft/PosterCraft-v1_RL", help='Custom transformer path')
    parser.add_argument('--qwen_model_path', type=str, default="Qwen/Qwen3-8B", help='Qwen model path')
    args = parser.parse_args()

    generator = PosterGenerator(
        pipeline_path=args.pipeline_path,
        custom_transformer_path=args.custom_transformer_path,
        qwen_model_path=args.qwen_model_path,
        device="cuda:0"  
    )
    
    image, final_prompt, seed = generator.generate(
        prompt=args.prompt,
        enable_recap=args.enable_recap,
        width=832,
        height=1216,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed  
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_poster_{timestamp}_{seed}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    print(f"Final prompt used: {final_prompt}")


if __name__ == "__main__":
    main()
