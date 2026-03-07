import os
import ollama
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "data", "maintenance_knowledge.txt")

def get_chat_recommendations(user_query: str, history: list = None):
    """
    Retrieval-Augmented Generation (RAG) implementation using local Ollama.
    """
    load_dotenv(override=True)
    model = os.environ.get("OLLAMA_MODEL", "llama3")

    # Load context
    try:
        with open(KNOWLEDGE_PATH, "r") as f:
            knowledge_base = f.read()
    except Exception as e:
        knowledge_base = "General solar maintenance documentation."

    # Construct the RAG prompt
    prompt = f"""
    Context (Technical Manual):
    {knowledge_base}

    Target Query: {user_query}

    Instructions:
    Answer based on the context above. If not in context, use general knowledge but mention it.
    Keep response under 80 words. Friendly and professional.
    """

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        error_msg = str(e).lower()
        print(f"Ollama RAG Error: {e}")
        
        # LOCAL FALLBACK
        normalized_query = user_query.lower()
        if "recommend" in normalized_query or "help" in normalized_query or "advice" in normalized_query:
            return (
                "Ollama is currently unavailable, but here are general maintenance recommendations from the manual:\n\n"
                "1. Check for voltage imbalances (< 2%).\n"
                "2. Ensure fans/vents are clear to prevent overheating (> 85°C).\n"
                "3. Verify grid frequency stability (50Hz/60Hz).\n"
                "4. Clean panels if power output is low despite clear skies."
            )
        
        return "The AI assistant is currently offline. Please ensure Ollama is running and the model is pulled."

