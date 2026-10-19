# Digital Twin on Groq

The twin uses Groq's [OpenAI-compatible API](https://console.groq.com/docs/openai)
with the existing `openai` Python dependency. No OpenAI API key is needed.

1. Install dependencies from the repository root:
   ```sh
   python -m pip install -r 1_foundations/twin/requirements.txt
   ```
2. Set `GROQ_API_KEY` in your environment or the repository's `.env` file.
   Get a key from [GroqCloud](https://console.groq.com/keys).
   Alternatively, copy this folder's `.env.example` to `.env` and fill it in.
3. Optionally set `GROQ_MODEL`. The default is `openai/gpt-oss-20b`.
   Choose a chat model with tool calling from [Groq's models](https://console.groq.com/docs/models).
4. Launch from the repository root:
   ```sh
   python 1_foundations/twin/app.py
   ```

The app also runs with `python app.py` from this folder. Environment variables
take precedence over `.env` values; the nearest `.env` to the app is loaded.
The profile is read from this folder's `linkedin.pdf` and `summary.txt`.

Configure `PUSHOVER_USER` and `PUSHOVER_TOKEN` for the existing contact and
unknown-question notification tools to deliver notifications.
