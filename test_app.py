import copy
import json
import os
import unittest
from unittest.mock import patch

import gradio as gr
import httpx
from openai import OpenAI

with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
    import app


class ChatHistoryTests(unittest.TestCase):
    def test_gradio_history_and_tool_round_trip(self):
        chatbot = gr.Chatbot()
        history = chatbot.preprocess(chatbot.postprocess([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "metadata": {"title": "Reply"}},
        ]))
        original_history = copy.deepcopy(history)
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload["messages"][1:3], [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ])
            for message in payload["messages"]:
                self.assertNotIn("metadata", message)
                self.assertNotIn("options", message)
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test", "type": "function", "function": {
                        "name": "record_unknown_question",
                        "arguments": '{"question":"Unknown question"}',
                    },
                }]}
                finish_reason = "tool_calls"
            else:
                self.assertEqual(payload["messages"][-1], {
                    "role": "tool", "content": '"OK"', "tool_call_id": "call_test",
                })
                message = {"role": "assistant", "content": "I do not know."}
                finish_reason = "stop"
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": app.MODEL_NAME, "choices": [{
                    "index": 0, "message": message, "finish_reason": finish_reason,
                }],
            })

        with OpenAI(
            api_key="test-key", base_url="https://api.groq.com/openai/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client, patch.object(app, "client", client), patch("tools.push") as push:
            self.assertEqual(app.chat("Unknown question", history), "I do not know.")
            push.assert_called_once()
        self.assertEqual(len(requests), 2)
        self.assertEqual(history, original_history)

    def test_plain_text_history(self):
        with patch.object(app.client.chat.completions, "create") as create:
            create.return_value.choices[0].finish_reason = "stop"
            create.return_value.choices[0].message.content = "Hello"
            for history in ([], [{"role": "user", "content": "Hi", "metadata": {}}]):
                self.assertEqual(app.chat("Hello", history), "Hello")
                messages = create.call_args.kwargs["messages"]
                self.assertTrue(all(set(message) == {"role", "content"} for message in messages))


if __name__ == "__main__":
    unittest.main()
