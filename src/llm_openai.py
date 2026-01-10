from __future__ import annotations
from typing import List, Dict, Any, Optional

import os

from .llm_client import LLMClient, LLMResponse
from .config import load_dotenv, getenv_any

class OpenAIChatCompletionsClient(LLMClient):
    """Minimal OpenAI client wrapper (optional).

    Not used by default toy experiments.

    Setup:
      - `pip install openai`
      - Put OPENAI_API_KEY=... in .env (recommended) or set env var
      - Optionally: OPENAI_BASE_URL=... (for proxies/self-hosted gateways)

    This wrapper attempts to use the newer SDK if available.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        force_json: bool = True,
        dotenv_path: str = ".env",
    ):
        load_dotenv(dotenv_path, override=False)

        self.model = model
        self.force_json = force_json
        self.api_key = api_key or getenv_any("OPENAI_API_KEY")
        self.base_url = base_url or getenv_any("OPENAI_BASE_URL")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required (put it in .env or environment).")

        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install openai` to use OpenAIChatCompletionsClient") from e

        self._openai = openai
        self._client = None

        # New SDK: from openai import OpenAI; client = OpenAI(api_key=...)
        if hasattr(openai, "OpenAI"):
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)  # type: ignore

    def _extract_usage(self, resp) -> Optional[Dict[str, Any]]:
        # Try to be resilient across SDK versions
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return None
            # new sdk: u has input_tokens/output_tokens/total_tokens
            out = {}
            for k in ["input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"]:
                v = getattr(u, k, None) if not isinstance(u, dict) else u.get(k)
                if v is not None:
                    out[k] = int(v)
            return out or None
        except Exception:
            return None

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        # We keep tool handling *outside* the OpenAI SDK layer in this repo.
        # The model is instructed (in src/llm_agent.py) to output a single JSON object per turn.
        usage = None
        if self._client is not None:
            kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            if self.force_json:
                # Newer SDKs support response_format; if not, it will throw and we fall back.
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._client.chat.completions.create(**kwargs)  # type: ignore
            except TypeError:
                # response_format not supported by this SDK version
                kwargs.pop("response_format", None)
                resp = self._client.chat.completions.create(**kwargs)  # type: ignore

            text = resp.choices[0].message.content or ""
            usage = self._extract_usage(resp)
        else:
            # legacy style
            kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=0,
                api_key=self.api_key,
            )
            resp = self._openai.ChatCompletion.create(**kwargs)  # type: ignore
            text = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage")

        return LLMResponse(text=text, tool_calls=[], usage=usage)
