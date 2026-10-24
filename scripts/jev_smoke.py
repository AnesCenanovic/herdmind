"""Minimal Jev test via OpenRouter's Decisions API.

Setup:
    uv add requests python-dotenv
Run:
    uv run jev_test.py
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env from the current folder

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "typesafe/jev-1.13"  # or "~typesafe/jev-latest"

payload = {
    "model": MODEL,
    # "state" is the context Jev reasons over
    "state": {
        "ticket": "My checkout page shows a blank screen after I click Pay. I have tried two browsers.",
    },
    # each question is answered independently, in one request
    "questions": {
        # NOUL: yes/no -> probability of yes
        "is_bug": {
            "type": "noul",
            "instructions": "Is the customer reporting a software defect?",
            "criteria": {
                "true": "The customer describes broken or unexpected product behavior.",
                "false": "The customer is asking a question or requesting a feature.",
            },
        },
        # CHOICE: pick one option -> probability per option
        "team": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": {
                "payments": "Checkout, billing, or payment processing issues.",
                "frontend": "Rendering, layout, or browser compatibility issues.",
                "account": "Login, permissions, or profile issues.",
            },
        },
    },
}

resp = requests.post(
    "https://openrouter.ai/api/alpha/decisions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=30,
)

print("HTTP status:", resp.status_code)
data = resp.json()
print(json.dumps(data, indent=2))

if resp.ok:
    answers = data["answers"]
    print("\n--- Summary ---")
    print(f"Is it a bug?  {answers['is_bug']['noul']:.0%} yes")
    print(f"Owning team:  {answers['team']['choice']} (confidence {answers['team']['confidence']})")
    print(f"Cost of this call: ${data['usage']['cost']}")