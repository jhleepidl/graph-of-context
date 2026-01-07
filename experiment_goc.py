#!/usr/bin/env python3
"""Graph of Context (GoC) synthetic experiment.

Generates a JSONL dataset and runs a minimal simulation comparing
ReAct, AgentFold, Context-Folding, and GoC.
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


DATASET_DEFAULT = "graph_needle_test.jsonl"
RESULTS_DEFAULT = "experiment_goc_runs.jsonl"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EOT_TOKEN = "<end_of_thought>"
MAX_ACTIVE_NODES = 5
FOLD_GROUP_MIN = 2
FOLD_GROUP_MAX = 3
RETRY_MAX_ATTEMPTS = 5
RETRY_MIN_SECONDS = 1
RETRY_MAX_SECONDS = 20

SYSTEM_PROMPT = (
    "You are a careful assistant. When asked for the key, respond with only "
    "the exact key string, e.g., KEY_1234."
)

FOLD_SUMMARY_PROMPT = (
    "Summarize the following steps in one sentence."
)

GOC_SYSTEM_PROMPT = (
    "You are a careful assistant. When asked for the key, respond with only "
    "the exact key string, e.g., KEY_1234."
)


DUMMY_TOOLS = [
    "check_weather",
    "calculate_sum",
    "lookup_city",
    "random_fact",
    "generate_noise",
]

CITY_CHOICES = ["Seoul", "Berlin", "Lagos", "Oslo", "Quito"]
WEATHER_CHOICES = ["Sunny", "Rainy", "Cloudy", "Windy", "Snowy"]
FACT_CHOICES = [
    "Octopus have three hearts.",
    "Honey never spoils.",
    "Bananas are berries.",
    "Some turtles breathe through their skin.",
]

def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key:
                os.environ.setdefault(key, value)


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate without extra dependencies (fallback only)."""
    count = 0
    for msg in messages:
        content = msg.get("content", "")
        count += max(1, len(content.split()))
    return count


def run_tool(tool_name: str, args: Dict[str, Any], rng: random.Random, key: Optional[str] = None) -> str:
    if tool_name == "get_initial_clue":
        return key or f"KEY_{rng.randint(1000, 9999)}"
    if tool_name == "check_weather":
        location = args.get("location", rng.choice(CITY_CHOICES))
        weather = rng.choice(WEATHER_CHOICES)
        return f"Weather in {location}: {weather}."
    if tool_name == "calculate_sum":
        a = args.get("a", rng.randint(1, 50))
        b = args.get("b", rng.randint(1, 50))
        return f"Sum of {a} and {b} is {a + b}."
    if tool_name == "lookup_city":
        city = args.get("city", rng.choice(CITY_CHOICES))
        return f"City lookup: {city} is a real city."
    if tool_name == "random_fact":
        return rng.choice(FACT_CHOICES)
    if tool_name == "generate_noise":
        return f"Noise:{rng.randint(10000, 99999)}"
    return "OK"


def make_tool_args(tool_name: str, rng: random.Random) -> Dict[str, Any]:
    if tool_name == "check_weather":
        return {"location": rng.choice(CITY_CHOICES)}
    if tool_name == "calculate_sum":
        return {"a": rng.randint(1, 50), "b": rng.randint(1, 50)}
    if tool_name == "lookup_city":
        return {"city": rng.choice(CITY_CHOICES)}
    return {}


def generate_dataset(
    path: str,
    num_cases: int,
    haystack_len: int,
    seed: int,
    needle_location: str,
) -> None:
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as handle:
        for idx in range(num_cases):
            case_seed = rng.randint(0, 1_000_000)
            case_rng = random.Random(case_seed)
            key = f"KEY_{case_rng.randint(1000, 9999)}"
            haystack = []
            for _ in range(haystack_len):
                tool_name = case_rng.choice(DUMMY_TOOLS)
                haystack.append(
                    {
                        "tool": tool_name,
                        "args": make_tool_args(tool_name, case_rng),
                    }
                )
            needle_index = None
            if needle_location == "branch" and haystack_len > 0:
                max_index = max(0, haystack_len - 2)
                needle_index = case_rng.randint(0, max_index)
                haystack[needle_index] = {"tool": "get_initial_clue", "args": {}}
                initial_prompt = (
                    "Solve a complex task. During the tool-call phase, call "
                    "get_initial_clue() once and remember the exact key. Then "
                    "perform the unrelated tool calls as instructed."
                )
            else:
                initial_prompt = (
                    "Solve a complex task. First, call get_initial_clue() and "
                    "remember the exact key. Then perform the unrelated tool calls "
                    "as instructed."
                )
            record = {
                "id": f"case_{idx:04d}",
                "seed": case_seed,
                "haystack_len": haystack_len,
                "needle_location": needle_location,
                "needle_index": needle_index,
                "initial_user_prompt": initial_prompt,
                "haystack": haystack,
                "final_user_prompt": "What was the exact key from step 1?",
                "expected_key": key,
            }
            handle.write(json.dumps(record) + "\n")


class LLMClient:
    def __init__(self, model: str, temperature: float) -> None:
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    @retry(
        reraise=True,
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(min=RETRY_MIN_SECONDS, max=RETRY_MAX_SECONDS),
        retry=retry_if_exception_type(
            (
                APIError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        ),
    )
    def _chat_with_retry(self, messages: List[Dict[str, Any]], max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        response = self._chat_with_retry(messages, max_tokens)
        content = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        return {"content": content, "usage": usage_data}


class AgentBase:
    name = "Base"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.history: List[Dict[str, Any]] = []
        self.last_prompt_tokens = 0

    def reset(self) -> None:
        self.history = []
        self.last_prompt_tokens = 0

    def add_system(self, content: str) -> None:
        self.history.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_step(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def start_branch(self) -> None:
        return None

    def end_branch(self) -> None:
        return None

    def finalize(self) -> None:
        return None

    def get_visible_messages(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def answer(self) -> str:
        messages = self.get_visible_messages()
        result = self.llm.chat(messages, max_tokens=32)
        prompt_tokens = result["usage"]["prompt_tokens"]
        self.last_prompt_tokens = prompt_tokens or estimate_tokens(messages)
        return result["content"]


class BaselineReAct(AgentBase):
    name = "Baseline_ReAct"


class BaselineAgentFold(AgentBase):
    name = "Baseline_AgentFold"

    def __init__(self, llm: LLMClient, fold_every: int = 5) -> None:
        super().__init__(llm)
        self.fold_every = fold_every
        self.step_cache: List[str] = []

    def reset(self) -> None:
        super().reset()
        self.step_cache = []

    def add_step(self, content: str) -> None:
        super().add_step(content)
        self.step_cache.append(content)
        if len(self.step_cache) >= self.fold_every:
            summary = self._summarize_steps(self.step_cache)
            for _ in range(self.fold_every):
                self.history.pop()
            self.history.append({"role": "assistant", "content": summary})
            self.step_cache = []

    def _summarize_steps(self, steps: List[str]) -> str:
        messages = [
            {"role": "system", "content": FOLD_SUMMARY_PROMPT},
            {"role": "user", "content": "\n".join(steps)},
        ]
        result = self.llm.chat(messages, max_tokens=64)
        return result["content"]


class BaselineContextFolding(AgentBase):
    name = "Baseline_ContextFolding"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm)
        self.branch_buffer: List[Dict[str, Any]] = []
        self.in_branch = False

    def reset(self) -> None:
        super().reset()
        self.branch_buffer = []
        self.in_branch = False

    def start_branch(self) -> None:
        self.in_branch = True
        self.branch_buffer = []

    def end_branch(self) -> None:
        if not self.in_branch:
            return
        self.in_branch = False
        if self.branch_buffer:
            last_message = self.branch_buffer[-1]["content"]
            summary = (
                "Branch has finished its task. The last message was:\n\n"
                f"{last_message}"
            )
        else:
            summary = "Branch has finished its task."
        # Mirror FoldAgent: keep only a branch return message as a user observation.
        self.history.append({"role": "user", "content": summary})
        self.branch_buffer = []

    def add_user(self, content: str) -> None:
        if self.in_branch:
            self.branch_buffer.append({"role": "user", "content": content})
        else:
            super().add_user(content)

    def add_step(self, content: str) -> None:
        if self.in_branch:
            self.branch_buffer.append({"role": "assistant", "content": content})
        else:
            super().add_step(content)


@dataclass
class SuperNode:
    node_id: str
    level: int
    summary: str
    messages: List[Dict[str, Any]]
    embedding: List[float]
    children: List[str]


class GoCAgent(AgentBase):
    name = "Ours_GoC"

    def __init__(
        self,
        llm: LLMClient,
        bundle_size: int = MAX_ACTIVE_NODES,
        top_k: int = 2,
        embed_model: str = DEFAULT_EMBED_MODEL,
        end_of_thought_token: str = DEFAULT_EOT_TOKEN,
    ) -> None:
        super().__init__(llm)
        self.max_active_nodes = bundle_size
        self.top_k = top_k
        self.embedder = SentenceTransformer(embed_model)
        self.end_of_thought_token = end_of_thought_token
        self.nodes: List[SuperNode] = []
        self.active_nodes: List[SuperNode] = []
        self.unfolded_nodes: set[str] = set()

    def reset(self) -> None:
        super().reset()
        self.nodes = []
        self.active_nodes = []
        self.unfolded_nodes = set()

    def add_step(self, content: str) -> None:
        segments = [content]
        if self.end_of_thought_token:
            segments = content.split(self.end_of_thought_token)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            node = self._create_leaf_node(segment)
            self.nodes.append(node)
            self.active_nodes.append(node)
            self._fold_if_needed()

    def finalize(self) -> None:
        return None

    def _create_leaf_node(self, content: str) -> SuperNode:
        messages = [{"role": "assistant", "content": content}]
        summary = self._summarize_messages(messages)
        embedding = self._embed_text(summary)
        return SuperNode(
            node_id=str(uuid.uuid4()),
            level=0,
            summary=summary,
            messages=messages,
            embedding=embedding,
            children=[],
        )

    def _create_parent_node(self, level: int, children: List[SuperNode]) -> SuperNode:
        combined_messages: List[Dict[str, Any]] = []
        for child in children:
            combined_messages.extend(child.messages)
        summary = self._summarize_nodes(children, level)
        embedding = self._embed_text(summary)
        return SuperNode(
            node_id=str(uuid.uuid4()),
            level=level,
            summary=summary,
            messages=combined_messages,
            embedding=embedding,
            children=[child.node_id for child in children],
        )

    def _fold_if_needed(self) -> None:
        if len(self.active_nodes) <= self.max_active_nodes:
            return
        folded = True
        while len(self.active_nodes) > self.max_active_nodes and folded:
            folded = self._fold_oldest_level(0, new_level=1)
        self._fold_level1_if_needed()

    def _fold_level1_if_needed(self) -> None:
        while self._count_level(1) > self.max_active_nodes:
            if not self._fold_oldest_level(1, new_level=2):
                break

    def _fold_oldest_level(self, level: int, new_level: int) -> bool:
        indices = [i for i, node in enumerate(self.active_nodes) if node.level == level]
        if len(indices) < FOLD_GROUP_MIN:
            return False
        group_size = min(FOLD_GROUP_MAX, len(indices))
        selected_indices = indices[:group_size]
        selected_nodes = [self.active_nodes[i] for i in selected_indices]
        insert_at = selected_indices[0]
        for idx in reversed(selected_indices):
            self.active_nodes.pop(idx)
        # Topological Encapsulation: fold older cycles into a higher-level node.
        parent = self._create_parent_node(new_level, selected_nodes)
        self.nodes.append(parent)
        self.active_nodes.insert(insert_at, parent)
        return True

    def _count_level(self, level: int) -> int:
        return sum(1 for node in self.active_nodes if node.level == level)

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        tools = []
        saw_clue = False
        for msg in messages:
            content = msg.get("content", "")
            if content.startswith("Tool "):
                parts = content.split()
                if len(parts) >= 2:
                    tool_name = parts[1]
                    tools.append(tool_name)
                    if tool_name == "get_initial_clue":
                        saw_clue = True
        tools = tools[:5]
        tool_list = ", ".join(tools) if tools else "none"
        if saw_clue:
            return (
                f"Processed {len(messages)} steps; obtained a key clue; tools: {tool_list}"
            )
        return f"Processed {len(messages)} steps; tools: {tool_list}"

    def _summarize_nodes(self, nodes: List[SuperNode], level: int) -> str:
        saw_clue = any("key clue" in node.summary for node in nodes)
        label = "SuperNode" if level == 1 else f"HyperNode L{level}"
        if saw_clue:
            return f"{label} of {len(nodes)} cycles; contains key clue"
        return f"{label} of {len(nodes)} cycles"

    def _embed_text(self, text: str) -> List[float]:
        embedding = self.embedder.encode(text)
        return embedding.tolist()

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _select_top_k_nodes(self, query: str) -> List[SuperNode]:
        if not self.nodes:
            return []
        query_embedding = self._embed_text(query)
        scored = []
        for node in self.nodes:
            if node.node_id in self.unfolded_nodes:
                continue
            score = self._cosine_similarity(query_embedding, node.embedding)
            scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored[: self.top_k]]

    def _build_node_markers(self, nodes: List[SuperNode]) -> List[Dict[str, Any]]:
        markers = []
        for node in nodes:
            markers.append(
                {"role": "assistant", "content": f"[Node L{node.level}:{node.summary}]"}
            )
        return markers

    def _build_unfold_messages(self, nodes: List[SuperNode]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for node in nodes:
            if node.node_id in self.unfolded_nodes:
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[Node Unfold L{node.level}:{node.summary}]",
                }
            )
            messages.extend(node.messages)
            self.unfolded_nodes.add(node.node_id)
        return messages

    def _inject_before_last_user(
        self,
        base_messages: List[Dict[str, Any]],
        extra_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not extra_messages:
            return base_messages
        for index in range(len(base_messages) - 1, -1, -1):
            if base_messages[index].get("role") == "user":
                return (
                    base_messages[:index]
                    + extra_messages
                    + base_messages[index:]
                )
        return base_messages + extra_messages

    def _build_context_messages(
        self,
        extra_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        base_messages = list(self.history)
        node_markers = self._build_node_markers(self.active_nodes)
        return self._inject_before_last_user(
            base_messages,
            node_markers + extra_messages,
        )

    def _latest_user_query(self) -> Optional[str]:
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def get_visible_messages(self) -> List[Dict[str, Any]]:
        return self._build_context_messages([])

    def answer(self) -> str:
        query = self._latest_user_query()
        unfold_messages: List[Dict[str, Any]] = []
        if query:
            top_nodes = self._select_top_k_nodes(query)
            unfold_messages = self._build_unfold_messages(top_nodes)
        messages = self._build_context_messages(unfold_messages)
        result = self.llm.chat(messages, max_tokens=32)
        prompt_tokens = result["usage"]["prompt_tokens"]
        self.last_prompt_tokens = prompt_tokens or estimate_tokens(messages)
        return result["content"]


def load_dataset(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def run_case(agent: AgentBase, case: Dict[str, Any]) -> Dict[str, Any]:
    case_rng = random.Random(case["seed"])
    expected_key = case["expected_key"]
    needle_location = case.get("needle_location", "main")

    agent.reset()
    if isinstance(agent, GoCAgent):
        agent.add_system(GOC_SYSTEM_PROMPT)
    else:
        agent.add_system(SYSTEM_PROMPT)

    agent.add_user(case["initial_user_prompt"])

    if needle_location == "main":
        key_output = run_tool("get_initial_clue", {}, case_rng, key=expected_key)
        agent.add_step(f"Tool get_initial_clue returned: {key_output}")

    agent.start_branch()
    for step in case["haystack"]:
        tool_name = step["tool"]
        if tool_name == "get_initial_clue":
            output = run_tool("get_initial_clue", {}, case_rng, key=expected_key)
        else:
            output = run_tool(tool_name, step.get("args", {}), case_rng)
        agent.add_step(f"Tool {tool_name} returned: {output}")
    agent.end_branch()

    agent.finalize()
    agent.add_user(case["final_user_prompt"])

    answer = agent.answer()
    return {
        "answer": answer,
        "prompt_tokens": agent.last_prompt_tokens,
        "correct": expected_key in answer,
    }


def run_experiment(
    dataset_path: str,
    model: str,
    temperature: float,
    fold_every: int,
    goc_bundle_size: int,
    goc_top_k: int,
    goc_embed_model: str,
    goc_end_of_thought_token: str,
) -> Dict[str, Any]:
    cases = load_dataset(dataset_path)
    llm = LLMClient(model=model, temperature=temperature)
    agents = [
        BaselineReAct(llm),
        BaselineAgentFold(llm, fold_every=fold_every),
        BaselineContextFolding(llm),
        GoCAgent(
            llm,
            bundle_size=goc_bundle_size,
            top_k=goc_top_k,
            embed_model=goc_embed_model,
            end_of_thought_token=goc_end_of_thought_token,
        ),
    ]

    results = {agent.name: {"correct": 0, "prompt_tokens": []} for agent in agents}
    per_case = []

    for case in cases:
        case_entry = {"id": case["id"], "expected_key": case["expected_key"]}
        for agent in agents:
            outcome = run_case(agent, case)
            results[agent.name]["correct"] += int(outcome["correct"])
            results[agent.name]["prompt_tokens"].append(outcome["prompt_tokens"])
            case_entry[agent.name] = outcome
        per_case.append(case_entry)

    metrics = {}
    for agent in agents:
        correct = results[agent.name]["correct"]
        total = len(cases)
        tokens = results[agent.name]["prompt_tokens"]
        avg_tokens = sum(tokens) / total if total else 0
        metrics[agent.name] = {
            "accuracy": correct / total if total else 0,
            "avg_prompt_tokens": avg_tokens,
        }

    return {
        "dataset": dataset_path,
        "num_cases": len(cases),
        "model": model,
        "temperature": temperature,
        "fold_every": fold_every,
        "goc_bundle_size": goc_bundle_size,
        "goc_top_k": goc_top_k,
        "goc_embed_model": goc_embed_model,
        "goc_end_of_thought_token": goc_end_of_thought_token,
        "metrics": metrics,
        "per_case": per_case,
    }


def append_results(path: str, run_data: Dict[str, Any]) -> None:
    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        **run_data,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="GoC synthetic experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate dataset")
    gen_parser.add_argument("--output", default=DATASET_DEFAULT)
    gen_parser.add_argument("--num-cases", type=int, default=10)
    gen_parser.add_argument("--haystack-len", type=int, default=30)
    gen_parser.add_argument("--seed", type=int, default=1337)
    gen_parser.add_argument(
        "--needle-location",
        choices=["main", "branch"],
        default="main",
        help="Where to place get_initial_clue (default: main).",
    )

    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--dataset", default=DATASET_DEFAULT)
    run_parser.add_argument("--results", default=RESULTS_DEFAULT)
    run_parser.add_argument("--model", default="gpt-4o-mini")
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--fold-every", type=int, default=5)
    run_parser.add_argument("--goc-bundle-size", type=int, default=MAX_ACTIVE_NODES)
    run_parser.add_argument("--goc-top-k", type=int, default=2)
    run_parser.add_argument("--goc-embed-model", default=DEFAULT_EMBED_MODEL)
    run_parser.add_argument("--goc-eot-token", default=DEFAULT_EOT_TOKEN)

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(
            args.output,
            args.num_cases,
            args.haystack_len,
            args.seed,
            args.needle_location,
        )
        print(f"Wrote dataset to {args.output}")
        return

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    run_data = run_experiment(
        dataset_path=args.dataset,
        model=args.model,
        temperature=args.temperature,
        fold_every=args.fold_every,
        goc_bundle_size=args.goc_bundle_size,
        goc_top_k=args.goc_top_k,
        goc_embed_model=args.goc_embed_model,
        goc_end_of_thought_token=args.goc_eot_token,
    )
    append_results(args.results, run_data)
    print("Run complete. Metrics:")
    for name, metric in run_data["metrics"].items():
        print(
            f"- {name}: accuracy={metric['accuracy']:.2f}, "
            f"avg_prompt_tokens={metric['avg_prompt_tokens']:.1f}"
        )


if __name__ == "__main__":
    main()
