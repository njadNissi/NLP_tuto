# model_utils.py
import os
from sentence_transformers import SentenceTransformer

def get_cached_model(
    model_name="all-MiniLM-L6-v2", 
    cache_dir="./model_cache"
):
    """Load SentenceTransformer from local cache (or download and cache if missing)."""
    # Create cache dir if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    # Replace slashes in model name to avoid directory issues (e.g., "sentence-transformers/all-MiniLM-L6-v2" → "sentence-transformers_all-MiniLM-L6-v2")
    safe_model_name = model_name.replace("/", "_")
    model_cache_path = os.path.join(cache_dir, safe_model_name)

    try:
        # Load from cache first
        if os.path.exists(model_cache_path):
            print(f"Loading cached model: {model_name} (path: {model_cache_path})")
            return SentenceTransformer(model_cache_path)
        
        # Download and save to cache if missing
        print(f"Downloading model: {model_name} (will cache for future use)")
        model = SentenceTransformer(model_name)
        model.save(model_cache_path)
        print(f"Model cached successfully at: {model_cache_path}")
        return model
    
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {str(e)}")
        # Fallback to lightweight default model
        fallback_model = "all-MiniLM-L6-v2"
        fallback_cache_path = os.path.join(cache_dir, fallback_model.replace("/", "_"))
        print(f"⚠️ Falling back to default model: {fallback_model}")
        
        if os.path.exists(fallback_cache_path):
            return SentenceTransformer(fallback_cache_path)
        # Download fallback if not cached
        fallback_model_obj = SentenceTransformer(fallback_model)
        fallback_model_obj.save(fallback_cache_path)
        return fallback_model_obj

def preload_model(model_name="all-MiniLM-L6-v2"):
    """Wrapper to preload model (for consistency across scripts)."""
    return get_cached_model(model_name)