"""Authored metadata fixtures; isolate the resolver without corpus or network imports."""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "lamurre_layer.py"


def resolver():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "REGION_ALIASES" for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "recording_region")
    ]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["recording_region"]


def test_documented_identifier_alias():
    region = resolver()
    assert region("varp_laptl1", {"varp_laptl": "LVÄ"}) == "LVÄ"


def test_exact_recording_metadata_wins_over_alias():
    region = resolver()
    assert region("varp_laptl1", {"varp_laptl1": "authored-exact", "varp_laptl": "authored-alias"}) == "authored-exact"


def test_unknown_recording_stays_unknown_without_guessing():
    region = resolver()
    assert region("authored_fixture2", {"authored_fixture": "LVÄ"}) == "?"
    assert region("varp_laptl1", {}) == "?"
