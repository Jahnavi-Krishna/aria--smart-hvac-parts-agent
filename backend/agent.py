import json
from datetime import date
from openai import OpenAI
import tools

client = OpenAI()

SYSTEM_PROMPT = """
You are a helpful, expert AI assistant specializing in helping 
customers find the right parts and solutions.

GROUNDING RULES — never break these:
- All part numbers, compatibility claims, and prices must come 
  from tool results. Never invent them.
- Never invent URLs. Only surface URLs from tool results.
- If data is missing, say so and ask for more information.
- Compatibility claims require check_compatibility tool call. No guessing.

RESPONSE RULES:
- Maximum 3 short paragraphs per response.
- Be warm, precise, and expert.
- Always end with: ||SUGGEST: option1 | option2 | option3

Today's date: {date}
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_parts",
            "description": "Search for parts by symptom, description, or keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string",
                                  "description": "What the user is looking for"},
                    "category":   {"type": "string",
                                  "description": "Optional category filter"}
                },
                "required": ["query_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_compatibility",
            "description": "Check if a part is compatible with a specific model number",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_number":  {"type": "string"},
                    "model_number": {"type": "string"}
                },
                "required": ["part_number", "model_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_guides",
            "description": "Find installation, troubleshooting, or policy guides",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent":      {"type": "string",
                                   "enum": ["installation",
                                            "troubleshooting",
                                            "policy"]},
                    "query_text":  {"type": "string"},
                    "part_number": {"type": "string"}
                },
                "required": ["intent", "query_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_part_by_number",
            "description": "Get exact part details by part number",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_number": {"type": "string"}
                },
                "required": ["part_number"]
            }
        }
    }
]

def run_agent(message: str, history: list,
              image_base64: str = None,
              image_mime:   str = None) -> dict:

    system = SYSTEM_PROMPT.format(date=date.today().isoformat())

    if image_base64:
        user_content = [
            {"type": "text", "text": message or "What do you see?"},
            {"type": "image_url",
             "image_url": {
                 "url": f"data:{image_mime};base64,{image_base64}"
             }}
        ]
    else:
        user_content = message

    messages = [{"role": "system", "content": system}]
    messages += history[-12:]
    messages.append({"role": "user", "content": user_content})

    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = tools.execute(tc.function.name, args)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result
            })

    raw = msg.content or ""

    suggestions = []
    if "||SUGGEST:" in raw:
        parts = raw.split("||SUGGEST:")
        raw   = parts[0].strip()
        chips = parts[1].strip().split("|")
        suggestions = [c.strip() for c in chips if c.strip()]

    return {"answer": raw, "suggestions": suggestions}