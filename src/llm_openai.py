from __future__ import annotations
from typing import List, Dict, Any, Optional

import os
import sys

from .llm_client import LLMClient, LLMResponse
from .config import load_dotenv, getenv_any


def _is_gpt5_family(model: str) -> bool:
    """Heuristic check for GPT-5 family model ids."""
    m = (model or "").lower().strip()
    return m.startswith("gpt-5")


_WARN_ONCE: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARN_ONCE:
        return
    _WARN_ONCE.add(key)
    print(msg, file=sys.stderr)


def _coerce_messages_for_responses(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Responses API accepts {role, content} messages; prefer developer over system for newer models."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "user")
        if role == "system":
            role = "developer"
        out.append({"role": role, "content": m.get("content") or ""})
    return out


def _extract_response_text(resp: Any) -> str:
    """Best-effort extraction of visible text from Responses API objects across SDK versions."""
    # New SDKs often expose output_text
    try:
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass

    parts: List[str] = []
    try:
        output = getattr(resp, "output", None)
        if not output:
            return ""
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                txt = getattr(c, "text", None)
                if isinstance(txt, str) and txt:
                    parts.append(txt)
    except Exception:
        return ""
    return "".join(parts)

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
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        force_json: bool = True,
        dotenv_path: str = ".env",
    ):
        load_dotenv(dotenv_path, override=False)

        self.model = model
        self.temperature = float(temperature)
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
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            # IMPORTANT: GPT-5 family models (e.g., gpt-5-mini) will error if sampling params
            # like temperature/top_p/logprobs are included (unless using specific GPT-5.2
            # settings). We defensively omit temperature for GPT-5 family models.
            # See: latest-model parameter compatibility docs.
            if _is_gpt5_family(self.model):
                if self.temperature not in (0.0, 0, None):
                    _warn_once(
                        "gpt5_no_temperature",
                        "[llm_openai] NOTE: GPT-5 family model detected; ignoring temperature (not supported for gpt-5-mini/nano/5 by default).",
                    )
            else:
                kwargs["temperature"] = self.temperature

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
                temperature=self.temperature,
                api_key=self.api_key,
            )
            resp = self._openai.ChatCompletion.create(**kwargs)  # type: ignore
            text = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage")

        return LLMResponse(text=text, tool_calls=[], usage=usage)

    def complete(self, messages: List[Dict[str, str]], force_json: bool = False, **kwargs) -> Dict[str, Any]:
        """Compatibility wrapper.

        Some runners call .complete(...) and expect a dict with keys:
          - text: str
          - usage: dict

        We route to .generate(...) and map the response.
        force_json controls JSON output formatting for *this call only*.
        """
        old_force_json = getattr(self, "force_json", True)
        self.force_json = bool(force_json)
        try:
            resp = self.generate(messages=messages)
            return {"text": resp.text, "usage": resp.usage or {}}
        finally:
            self.force_json = old_force_json


class OpenAIResponsesClient(LLMClient):
    """OpenAI Responses API wrapper.

    Recommended for GPT-5 family reasoning models.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        force_json: bool = False,
        dotenv_path: str = ".env",
    ):
        load_dotenv(dotenv_path, override=False)

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_output_tokens = max_output_tokens
        self.force_json = force_json

        self.api_key = api_key or getenv_any("OPENAI_API_KEY")
        self.base_url = base_url or getenv_any("OPENAI_BASE_URL")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required (put it in .env or environment).")

        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install openai` to use OpenAIResponsesClient") from e

        if not hasattr(openai, "OpenAI"):
            raise RuntimeError(
                "OpenAIResponsesClient requires a recent openai Python SDK (OpenAI class). "
                "Try: pip install -U openai"
            )

        self._openai = openai
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)  # type: ignore

    def _extract_usage(self, resp) -> Optional[Dict[str, Any]]:
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return None
            out: Dict[str, Any] = {}
            for k in ["input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"]:
                v = getattr(u, k, None) if not isinstance(u, dict) else u.get(k)
                if v is not None:
                    out[k] = int(v)
            return out or None
        except Exception:
            return None

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        # We do not pass sampling params (temperature/top_p) here; GPT-5 family models
        # generally reject them unless using GPT-5.2 with reasoning.effort="none".
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": _coerce_messages_for_responses(messages),
            "text": {"format": {"type": "text"}},
        }
        if self.verbosity:
            kwargs["text"]["verbosity"] = self.verbosity
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(self.max_output_tokens)

        # JSON mode (best effort) when requested.
        if self.force_json:
            # In Responses API, JSON mode can be achieved with a JSON schema or SDK helpers;
            # for backward compatibility we use json_object when available.
            kwargs["text"]["format"] = {"type": "json_object"}

        resp = self._client.responses.create(**kwargs)  # type: ignore
        text = _extract_response_text(resp)
        usage = self._extract_usage(resp)
        return LLMResponse(text=text or "", tool_calls=[], usage=usage)

    def complete(self, messages: List[Dict[str, str]], force_json: bool = False, **kwargs) -> Dict[str, Any]:
        old_force_json = getattr(self, "force_json", False)
        self.force_json = bool(force_json)
        try:
            resp = self.generate(messages=messages)
            return {"text": resp.text, "usage": resp.usage or {}}
        finally:
            self.force_json = old_force_json


def make_openai_client(
    *,
    model: str,
    temperature: float = 0.0,
    api_mode: str = "auto",
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    force_json: bool = False,
    dotenv_path: str = ".env",
) -> LLMClient:
    """Factory that chooses Chat Completions vs Responses API.

    - For GPT-5 family models, Responses API is the default in auto mode.
    - For older models, Chat Completions is the default.
    """
    mode = (api_mode or "auto").lower().strip()
    if mode not in {"auto", "chat", "responses"}:
        raise ValueError(f"api_mode must be auto|chat|responses, got: {api_mode}")

    if mode == "auto":
        mode = "responses" if _is_gpt5_family(model) else "chat"

    if mode == "responses":
        return OpenAIResponsesClient(
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            max_output_tokens=max_output_tokens,
            force_json=force_json,
            dotenv_path=dotenv_path,
        )

    return OpenAIChatCompletionsClient(
        model=model,
        temperature=temperature,
        force_json=force_json,
        dotenv_path=dotenv_path,
    )

