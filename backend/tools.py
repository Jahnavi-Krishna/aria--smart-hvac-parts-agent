import json
from rag import search_parts, search_guides

def tool_search_parts(query_text: str, category: str = None) -> dict:
    # TRADEOFF: semantic search finds meaning but misses exact IDs.
    # Use tool_get_part_by_number for exact part number lookups.
    results = search_parts(query_text)
    return {"results": results, "count": len(results)}

def tool_search_guides(intent: str, query_text: str,
                       part_number: str = None) -> dict:
    # intent: "installation" | "troubleshooting" | "policy"
    results = search_guides(query_text, guide_type=intent)
    return {"results": results, "count": len(results)}

def tool_check_compatibility(part_number: str, model_number: str) -> dict:
    # TRADEOFF: deterministic lookup only — no LLM inference.
    # Production: query live compatibility DB.
    try:
        with open("data/compatibility.json") as f:
            data = json.load(f)
        for entry in data:
            if (str(entry['part_number']) == str(part_number) and
                    entry['model_number'].upper() == model_number.upper()):
                return {"compatible": entry['compatible'],
                        "notes": entry.get('notes', '')}
        return {"compatible": None,
                "notes": "No compatibility data found for this combination."}
    except Exception as e:
        return {"error": str(e)}

def tool_get_part_by_number(part_number: str) -> dict:
    try:
        with open("data/products.json") as f:
            parts = json.load(f)
        for p in parts:
            if str(p['part_number']) == str(part_number):
                return p
        return {"error": f"Part {part_number} not found in local catalog."}
    except Exception as e:
        return {"error": str(e)}

# ── DISPATCHER ──────────────────────────────────
TOOL_MAP = {
    "search_parts":        tool_search_parts,
    "search_guides":       tool_search_guides,
    "check_compatibility": tool_check_compatibility,
    "get_part_by_number":  tool_get_part_by_number,
}

def execute(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})