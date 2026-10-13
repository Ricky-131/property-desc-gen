import os
import json
import csv
import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List of models to try in order — each has its own quota pool on Gemini free tier!
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

PROMPT_STYLE_LABELS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "role_based": "Role-Based",
}


def build_combined_prompt(prop: dict) -> str:
    features = ", ".join(prop.get("features", [])) or "N/A"
    bedrooms = int(prop.get("bedrooms", 0))
    bed_str = "Studio" if bedrooms == 0 else f"{bedrooms} bedroom(s)"
    price = f"${int(prop['price']):,}"
    area = f"{prop.get('area_sqft', 'N/A')} sq ft"

    prop_block = (
        f"Property Name : {prop['name']}\n"
        f"Type          : {prop['type']}\n"
        f"Location      : {prop['location']}\n"
        f"Bedrooms      : {bed_str}\n"
        f"Bathrooms     : {prop.get('bathrooms', 1)}\n"
        f"Area          : {area}\n"
        f"Price         : {price}\n"
        f"Key Features  : {features}"
    )

    few_shot_examples = """\
Example 1 —
Property: Cozy 2-bed, 1-bath cottage in Portland, OR. 750 sq ft. Garden and fireplace.
Description: Nestled in the heart of Portland, this charming 2-bedroom cottage offers a warm retreat from the city buzz. The inviting fireplace creates a cosy atmosphere on cool evenings, while the private garden is perfect for morning coffees. At 750 sq ft, it packs everything you need into a delightfully compact package. Priced to sell — a rare find in today's market.

Example 2 —
Property: Modern 3-bed, 2-bath condo in Miami, FL. 1,800 sq ft. Pool, gym, ocean view.
Description: Wake up to breathtaking ocean views every morning in this sleek 3-bedroom Miami condo. Spanning 1,800 sq ft, the open-plan layout floods with natural light and flows effortlessly onto a private balcony. Residents enjoy resort-style amenities including a rooftop pool and fully equipped gym. Sophisticated coastal living awaits.\
"""

    return f"""You are a real estate copywriting and evaluation expert. Complete the task below in one response.

## Property Details
{prop_block}

## Task
Generate THREE property listing descriptions for the property above — one per prompt style — then score all three.
Every description must be exactly 3–4 sentences.

### Style definitions
- ZERO_SHOT  : Write a compelling, professional real estate description with no examples given.
- FEW_SHOT   : Write a similar description inspired by these two examples:
{few_shot_examples}
- ROLE_BASED : You are an award-winning luxury real estate copywriter with 20 years of experience. Your descriptions are vivid, emotionally compelling and highlight the lifestyle, not just the features.

## Scoring (after writing all three)
Score each description on:
- accuracy (1–5): Does it correctly reflect the property details?
- tone     (1–5): Is it professional, engaging and appropriate?
- length   (1–5): Does it have exactly 3–4 sentences?

## Output format
Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "zero_shot": {{
    "description": "<3-4 sentence description>",
    "scores": {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": "<one sentence>"}}
  }},
  "few_shot": {{
    "description": "<3-4 sentence description>",
    "scores": {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": "<one sentence>"}}
  }},
  "role_based": {{
    "description": "<3-4 sentence description>",
    "scores": {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": "<one sentence>"}}
  }}
}}"""


def process_property(prop: dict) -> dict:
    """Generate + score all 3 styles in a single API call with model fallback."""
    prompt = build_combined_prompt(prop)

    last_error = None
    data = None

    for model_name in CANDIDATE_MODELS:
        try:
            print(f"Calling Gemini model: {model_name}...", flush=True)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Strip markdown code fences if present
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)
            print(f"Successfully generated using {model_name}!", flush=True)
            break
        except Exception as e:
            print(f"Model {model_name} failed: {e}. Trying next candidate...", flush=True)
            last_error = e

    if data is None:
        raise RuntimeError(f"All candidate models failed. Last error: {last_error}")

    # Add totals
    result = {"property": prop["name"], "styles": {}}
    for style in ["zero_shot", "few_shot", "role_based"]:
        s = data[style]["scores"]
        s["total"] = s["accuracy"] + s["tone"] + s["length"]
        result["styles"][style] = {
            "description": data[style]["description"],
            "scores": s,
        }
    return result


# ─────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────

def save_results(results: list[dict], base_name: str = None):
    if base_name is None:
        base_name = datetime.datetime.now().strftime("results_%Y%m%d_%H%M%S")
    json_path = f"{base_name}.json"
    csv_path = f"{base_name}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "style", "accuracy", "tone", "length", "total", "reasoning", "description"])
        for entry in results:
            for style in ["zero_shot", "few_shot", "role_based"]:
                data = entry["styles"][style]
                s = data["scores"]
                writer.writerow([
                    entry["property"], style,
                    s["accuracy"], s["tone"], s["length"], s["total"],
                    s["reasoning"], data["description"],
                ])
    return json_path, csv_path
