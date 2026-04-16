import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from deep_translator import GoogleTranslator

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
_tokenizer = None
_model = None
_device = None
_embed_cache = {}

def embed_action(action: str) -> np.ndarray:
    if action not in _embed_cache:
        eng = translate_to_english(action)
        _embed_cache[action] = normalize_vec(encode_text(eng))
    return _embed_cache[action]

def ensure_model():
    global _tokenizer, _model, _device
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(_device)

def encode_text(text: str) -> np.ndarray:
    ensure_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True).to(_device)
    with torch.no_grad():
        out = _model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text