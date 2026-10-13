import os
import json
import csv
import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.8-flash")

# ─────────────────────────────────────────────
# Prompt builders (from user-supplied property dict)
# ─────────────────────────────────────────────

def build_zero_shot_prompt(prop: dict) -> str:
    features = ", ".join(prop.get("features", []))
    bedrooms = prop.get("bedrooms", 0)
    bed_str = "Studio" if int(bedrooms) == 0 else f"{bedrooms} bedroom(s)"
    return (
        f"Write a property listing description for the following real estate property.\n\n"
        f"Property Name: {prop['name']}\n"
        f"Type: {prop['type']}\n"
        f"Location: {prop['location']}\n"
        f"Bedrooms: {bed_str}\n"
        f"Bathrooms: {prop.get('bathrooms', 1)}\n"
        f"Area: {prop.get('area_sqft', 'N/A')} sq ft\n"
        f"Price: ${int(prop['price']):,}\n"
        f"Key Features: {features}\n\n"
        f"Write a compelling, professional real estate description in 3-4 sentences."
    )


def build_few_shot_prompt(prop: dict) -> str:
    features = ", ".join(prop.get("features", []))
    bedrooms = prop.get("bedrooms", 0)
    bed_str = "Studio" if int(bedrooms) == 0 else f"{bedrooms} bedroom(s)"
    examples = """\
Here are two examples of great property descriptions:

Example 1:
Property: Cozy 2-bed, 1-bath cottage in Portland, OR. 750 sq ft. Garden and fireplace.
Description: Nestled in the heart of Portland, this charming 2-bedroom cottage offers a warm retreat from the city buzz. The inviting fireplace creates a cosy atmosphere on cool evenings, while the private garden is perfect for morning coffees. At 750 sq ft, it packs everything you need into a delightfully compact package. Priced to sell — a rare find in today's market.

Example 2:
Property: Modern 3-bed, 2-bath condo in Miami, FL. 1,800 sq ft. Pool, gym, ocean view.
Description: Wake up to breathtaking ocean views every morning in this sleek 3-bedroom Miami condo. Spanning 1,800 sq ft, the open-plan layout floods with natural light and flows effortlessly onto a private balcony. Residents enjoy resort-style amenities including a rooftop pool and fully equipped gym. Sophisticated coastal living awaits.

Now write a similar description for this property:
"""
    return (
        examples
        + f"Property Name: {prop['name']}\n"
        f"Type: {prop['type']}\n"
        f"Location: {prop['location']}\n"
        f"Bedrooms: {bed_str}, Bathrooms: {prop.get('bathrooms', 1)}, Area: {prop.get('area_sqft', 'N/A')} sq ft\n"
        f"Price: ${int(prop['price']):,}\n"
        f"Key Features: {features}\n\n"
        f"Description (3-4 sentences, professional and engaging):"
    )


def build_role_based_prompt(prop: dict) -> str:
    features = ", ".join(prop.get("features", []))
    bedrooms = prop.get("bedrooms", 0)
    bed_str = "Studio" if int(bedrooms) == 0 else f"{bedrooms} bedroom(s)"
    return (
        f"You are an award-winning luxury real estate copywriter with 20 years of experience. "
        f"Your descriptions are known for being vivid, emotionally compelling, and highly persuasive. "
        f"You always highlight the lifestyle a property offers, not just its features. "
        f"Write a premium listing description for the property below in exactly 3-4 sentences.\n\n"
        f"Property Name: {prop['name']}\n"
        f"Type: {prop['type']}\n"
        f"Location: {prop['location']}\n"
        f"Bedrooms: {bed_str}\n"
        f"Bathrooms: {prop.get('bathrooms', 1)}\n"
        f"Area: {prop.get('area_sqft', 'N/A')} sq ft\n"
        f"Price: ${int(prop['price']):,}\n"
        f"Key Features: {features}"
    )


PROMPT_BUILDERS = {
    "zero_shot":  ("Zero-Shot",   build_zero_shot_prompt),
    "few_shot":   ("Few-Shot",    build_few_shot_prompt),
    "role_based": ("Role-Based",  build_role_based_prompt),
}

# ─────────────────────────────────────────────
# Single combined scoring call (1 call for all 3)
# ─────────────────────────────────────────────

def score_all_descriptions(prop: dict, descriptions: dict) -> dict:
    """
    One Gemini call to score all 3 descriptions at once.
    Returns { "zero_shot": {...}, "few_shot": {...}, "role_based": {...} }
    """
    prop_summary = (
        f"Name: {prop['name']} | Type: {prop['type']} | Location: {prop['location']} | "
        f"Bedrooms: {prop.get('bedrooms', 0)} | Bathrooms: {prop.get('bathrooms', 1)} | "
        f"Area: {prop.get('area_sqft', 'N/A')} sq ft | Price: ${int(prop['price']):,} | "
        f"Features: {', '.join(prop.get('features', []))}"
    )

    desc_block = "\n\n".join(
        f"[{style.upper()}]\n{descriptions[style]}"
        for style in ["zero_shot", "few_shot", "role_based"]
    )

    scoring_prompt = f"""
You are a real estate content evaluator. Score the three descriptions below on three criteria each.
Return ONLY a valid JSON object — no markdown, no code fences, no explanation.

Property Data:
{prop_summary}

Descriptions to evaluate:
{desc_block}

Scoring Criteria (each scored 1–5):
- accuracy  : Does the description correctly reflect the property details?
- tone      : Is the tone professional, engaging and appropriate for a real estate listing?
- length    : Does the description have 3-4 sentences as instructed?

Return this exact JSON (fill in the numbers and reasoning strings):
{{
  "zero_shot":  {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": ""}},
  "few_shot":   {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": ""}},
  "role_based": {{"accuracy": 0, "tone": 0, "length": 0, "reasoning": ""}}
}}
"""
    response = model.generate_content(scoring_prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    scores = json.loads(text.strip())
    # Add computed totals
    for style in scores:
        s = scores[style]
        s["total"] = s["accuracy"] + s["tone"] + s["length"]
    return scores


# ─────────────────────────────────────────────
# Core: process one property (4 API calls total)
# ─────────────────────────────────────────────

def process_property(prop: dict) -> dict:
    """Generate descriptions with all 3 styles + score in a single batch. 4 API calls."""
    descriptions = {}
    for style, (label, builder) in PROMPT_BUILDERS.items():
        prompt = builder(prop)
        response = model.generate_content(prompt)
        descriptions[style] = response.text.strip()

    scores = score_all_descriptions(prop, descriptions)

    result = {"property": prop["name"], "styles": {}}
    for style in PROMPT_BUILDERS:
        result["styles"][style] = {
            "description": descriptions[style],
            "scores": scores[style],
        }
    return result


# ─────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────

def save_results(results: list[dict], base_name: str = None):
    if base_name is None:
        base_name = datetime.datetime.now().strftime("results_%Y%m%d_%H%M%S")
    json_path = f"{base_name}.json"
    csv_path  = f"{base_name}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "style", "accuracy", "tone", "length", "total", "reasoning", "description"])
        for entry in results:
            for style in PROMPT_BUILDERS:
                data = entry["styles"][style]
                s = data["scores"]
                writer.writerow([
                    entry["property"], style,
                    s["accuracy"], s["tone"], s["length"], s["total"],
                    s["reasoning"], data["description"],
                ])
    return json_path, csv_path
