# ast_utils.py

def normalize_bool_ast(ast):
    """Flatten nested AND/OR trees into lists."""
    if not isinstance(ast, dict):
        return ast

    if ast.get("op") in ("and","or"):
        op = ast["op"]
        clauses = []

        def collect(node):
            if isinstance(node, dict) and node.get("op") == op:
                collect(node["left"])
                collect(node["right"])
            else:
                clauses.append(normalize_bool_ast(node))

        collect(ast)
        return {"op":op, "clauses":clauses}

    if ast.get("op") == "not":
        return {"op":"not", "arg":normalize_bool_ast(ast["arg"])}

    return ast
