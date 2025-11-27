import importlib

class LocalGraphFallback:
    def __init__(self):
        self._adj = {}
    def seed(self, m):
        self._adj.update(m)
    def get_neighbors(self, node):
        return self._adj.get(node, [])

def load_graph():
    try:
        mod = importlib.import_module('incoming.graph_from_B')
        if hasattr(mod, 'get_neighbors'):
            return mod
        if hasattr(mod, 'neighbors'):
            return mod
        if hasattr(mod, 'GraphProvider'):
            return mod.GraphProvider()
    except Exception:
        pass
    return LocalGraphFallback()

graph = load_graph()