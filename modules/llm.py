import os

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Make sure your .env file exists "
        "and contains a valid Groq API key."
    )

client = Groq(api_key=API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

NO_ANSWER_MESSAGE = (
    "I could not find this information in the uploaded document."
)

SYSTEM_PROMPT = f"""You are an AI PDF Question Answering Assistant.

Your job is to answer questions ONLY using the information provided
in the document context given by the user.

Rules:
1. Use ONLY the provided document context to form your answer.
2. Do NOT use your own outside knowledge.
3. Do NOT make up or infer information that isn't in the context.
4. If the answer is not present in the context, reply exactly:
"{NO_ANSWER_MESSAGE}"
5. Use the conversation history ONLY to resolve references such as
"he", "she", "it", "that", or "their" in the current question.
Never treat the conversation history itself as a source of facts.
6. Be clear and concise. Summarize relevant information rather than
copying it verbatim where possible.
"""


def generate_answer(question, context, chat_history=""):

    # No retrieved context at all -- don't bother calling the LLM,
    # the answer is already known.
    if not context or not context.strip():
        return NO_ANSWER_MESSAGE

    user_prompt = f"""Conversation History:
{chat_history if chat_history.strip() else "(none)"}

Document Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    except RateLimitError:
        return (
            "⚠️ The AI service is currently rate-limited. "
            "Please wait a moment and try again."
        )

    except APIConnectionError:
        return (
            "⚠️ Could not connect to the AI service. "
            "Please check your internet connection and try again."
        )

    except APIError as e:
        return f"⚠️ The AI service returned an error: {e}"

    except Exception as e:
        return f"⚠️ An unexpected error occurred while generating the answer: {e}"