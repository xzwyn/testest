import json
from src.clients.azure_client import chat  

def evaluate_translation_pair(eng_text: str, ger_text: str, model_name=None):
    prompt = f"""
## ROLE
You are the Primary Translation Auditor for EN→DE corporate reports.
...
(The rest of your detailed prompt for Agent 1 is unchanged)
...
<German Translation>
{ger_text}
</German Translation>

## YOUR RESPONSE
Return the JSON object only—no extra text, no markdown.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        if j0 != -1 and j1 != -1:
            return json.loads(content[j0:j1])
        return {"error_type": "System Error",
                "explanation": "No JSON object in LLM reply."}
    except Exception as exc:
        print(f"evaluate_translation_pair → {exc}")
        return {"error_type": "System Error", "explanation": str(exc)}

def check_context_mismatch(eng_text: str, ger_text: str, model_name: str = None):
    prompt = f"""
ROLE: Narrative-Integrity Analyst
...
(The rest of your detailed prompt for Agent 3 is unchanged)
...
<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"context_match": "Error", "explanation": str(exc)}