from __future__ import annotations

import unittest

from hospital_service.service.llm_client import OpenAICompatibleChatClient


class OpenAICompatibleChatClientTest(unittest.TestCase):
    def test_request_payload_disables_thinking_and_streaming(self) -> None:
        client = OpenAICompatibleChatClient(
            model_api="https://model.example/v1",
            api_key="secret",
            model="configured-model",
            timeout_seconds=30,
        )

        payload = client._request_payload(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            extra_body={"response_format": {"type": "json_object"}},
        )

        self.assertIs(payload["enable_thinking"], False)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["model"], "configured-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_glm_request_uses_supported_thinking_control(self) -> None:
        client = OpenAICompatibleChatClient(
            model_api="https://model.example/v1",
            api_key="secret",
            model="glm-5.2-fast",
            timeout_seconds=30,
        )

        payload = client._request_payload(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            extra_body={"enable_thinking": True, "stream": True},
        )

        self.assertNotIn("enable_thinking", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIs(payload["stream"], False)

    def test_kimi_request_uses_supported_thinking_control(self) -> None:
        client = OpenAICompatibleChatClient(
            model_api="https://model.example/v1",
            api_key="secret",
            model="kimi-k2.6-fast",
            timeout_seconds=30,
        )

        payload = client._request_payload(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            extra_body={"enable_thinking": True, "thinking": {"type": "enabled"}},
        )

        self.assertNotIn("enable_thinking", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIs(payload["stream"], False)

    def test_deepseek_request_uses_supported_thinking_control(self) -> None:
        client = OpenAICompatibleChatClient(
            model_api="https://model.example/v1",
            api_key="secret",
            model="deepseek-v4-flash-0731",
            timeout_seconds=30,
        )

        payload = client._request_payload(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            extra_body={"enable_thinking": True},
        )

        self.assertNotIn("enable_thinking", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIs(payload["stream"], False)


if __name__ == "__main__":
    unittest.main()
