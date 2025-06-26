from .nodes import LoadPosterCraftPrompt, LoadPipeline, LoadCustomTransformer, LoadQwenModel, PosterCraft, SavePosterCraft

NODE_CLASS_MAPPINGS = {
    "LoadPosterCraftPrompt": LoadPosterCraftPrompt,
    "LoadPipeline": LoadPipeline,
    "LoadCustomTransformer": LoadCustomTransformer,
    "LoadQwenModel": LoadQwenModel,
    "PosterCraft": PosterCraft,
    "SavePosterCraft": SavePosterCraft,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPosterCraftPrompt": "Load PosterCraft Prompt",
    "LoadPipeline": "Load Pipeline",
    "LoadCustomTransformer": "Load Custom Transformer",
    "LoadQwenModel": "Load Qwen Model",
    "PosterCraft": "PosterCraft",
    "SavePosterCraft": "Save PosterCraft",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
