import importlib

def _load_memberA_embedder():
    try:
        mod = importlib.import_module('incoming.embedder_from_A')
        if hasattr(mod, 'embed'):
            return mod.embed
        if hasattr(mod, 'Embedder'):
            return mod.Embedder().embed
    except Exception:
        return None

def _load_memberB_embeddings():
    try:
        mod = importlib.import_module('core.embeddings')
        if hasattr(mod, 'Embeddings'):
            return mod.Embeddings().embed
    except Exception:
        return None

_embed = _load_memberA_embedder() or _load_memberB_embeddings()
if _embed is None:
    def _dummy_embed(text: str):
        return [float(ord(c) % 10)/10.0 for c in text[:128]]
    _embed = _dummy_embed

def embed(text: str):
    return _embed(text)