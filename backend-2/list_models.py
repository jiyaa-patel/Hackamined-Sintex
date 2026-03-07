import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    api_key = api_key.strip('"').strip("'")

print(f"DEBUG: API Key from env: '{api_key}'")
client = genai.Client(api_key=api_key)

try:
    print("Listing models...")
    for model in client.models.list():
        if 'generateContent' in model.supported_actions:
            print(f"- {model.name}")
except Exception as e:
    print(f"Error: {e}")
