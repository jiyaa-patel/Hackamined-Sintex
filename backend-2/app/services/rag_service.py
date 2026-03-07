import os
import time
import random
from google import genai
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "data", "maintenance_knowledge.txt")

def get_chat_recommendations(user_query: str, history: list = None):
    """
    Retrieval-Augmented Generation (RAG) implementation using modern Google GenAI SDK.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        api_key = api_key.strip('"').strip("'")
    
    if not api_key or not api_key.startswith("AIza"):
        return "I need a valid Google Gemini API key to assist you. Please update your .env file."

    # 1. Load context
    try:
        with open(KNOWLEDGE_PATH, "r") as f:
            knowledge_base = f.read()
    except Exception as e:
        knowledge_base = "General solar maintenance documentation."

    client = genai.Client(api_key=api_key, http_options={'timeout': 10.0})

    # 2. Construct the RAG prompt
    prompt = f"""
    Context (Technical Manual):
    {knowledge_base}

    Target Query: {user_query}

    Instructions:
    Answer based on the context above. If not in context, use general knowledge but mention it.
    Keep response under 80 words. Friendly and professional.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e).lower()
            print(f"RAG SDK Error (Attempt {attempt+1}): {e}")
            if "429" in error_msg or "exhausted" in error_msg or "quota" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    time.sleep(wait_time)
                    continue
                return "The AI assistant is currently receiving too many requests (API rate limit exceeded). Please wait a minute and try again."
            break
            
    return "I am currently unable to access my knowledge base. Please try again later."
