import os

from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    raise ValueError("Set GROQ_API_KEY in your environment or .env file before starting the twin.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def chat(message, history):
    # Gradio history includes UI metadata and normalized text blocks. Send only
    # the role and plain text expected by the chat API.
    chat_history = []
    for entry in history:
        content = entry["content"]
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part["type"] == "text")
        chat_history.append({"role": entry["role"], "content": content})
    messages = system + chat_history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
