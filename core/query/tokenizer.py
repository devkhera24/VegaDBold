# tokenizer.py
import re
from typing import List, NamedTuple

TOKEN_SPEC = [
    ("NUMBER",   r"\d+\.?\d*"),
    ("STRING",   r'"(?:\\.|[^"\\])*"'),
    ("COMMA",    r","),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("OP",       r"<=|>=|!=|=|<|>"),
    ("KW",       r"\b(FIND|NODES|WHERE|RETURN|GET|NODE|PROPERTIES|SEARCH|VECTOR|"
                 r"EMBED|K|FILTER|LIMIT|OFFSET|ORDER|BY|ASC|DESC|PATH|FROM|TO|"
                 r"MAXHOPS|DIRECTED|UNDIRECTED|CONTAINS|EXISTS|AND|OR|NOT)\b"),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_\.]*"),
    ("WS",       r"\s+"),
    ("UNKNOWN",  r".")
]

TOK_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))

class Token(NamedTuple):
    type: str
    value: str
    pos: int

def tokenize(text: str) -> List[Token]:
    tokens = []
    for m in TOK_REGEX.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind == "WS":
            continue
        tokens.append(Token(kind, val, m.start()))
    return tokens
