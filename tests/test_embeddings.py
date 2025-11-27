from core.embeddings import Embeddings

def test_embedding_basic_properties():
    emb = Embeddings()
    v = emb.embed("hello world")

    assert isinstance(v, list)
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)