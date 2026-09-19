import base64
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

# Groq's vision-capable model. qwen/qwen3.6-27b is Groq's other
# documented vision option, but it's a preview model not enabled on
# every account -- qwen3.8-27b was confirmed available on this key.
# If this ever 404s again, run client.models.list() (see llm.py's
# original setup) to check what's currently available and update
# this constant.
VISION_MODEL_NAME = "qwen/qwen3.8-27b"

DEFAULT_PROMPT = (
    "Describe this image in detail. If it contains a diagram, chart, "
    "graph, or table, explain what it shows, including any labels, "
    "axes, values, or relationships depicted. If it contains visible "
    "text, include that text as well. Be thorough but concise."
)


def describe_image(image_bytes, prompt=DEFAULT_PROMPT, mime_type="image/png"):
    """
    Send an image to a vision-capable Groq model and return a text
    description of its contents. Used for:
    - Diagrams/charts embedded in PDF pages
    - Directly uploaded image files (JPG/PNG)

    Returns a plain string -- either the description, or a short
    user-facing error message (never raises), so callers can safely
    drop the result straight into document text.
    """

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{base64_image}"

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            temperature=0.2,
            max_tokens=600,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except RateLimitError:
        return (
            "⚠️ Image analysis is rate-limited right now. "
            "Please try again shortly."
        )

    except APIConnectionError:
        return "⚠️ Could not connect to the vision service."

    except APIError as e:
        return f"⚠️ Vision service error: {e}"

    except Exception as e:
        return f"⚠️ Unexpected error analyzing image: {e}"