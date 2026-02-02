"""Offline smoke test for LLM-declared dependency ingestion (GoC).

This does NOT call any LLM. It simulates two tool calls and injects a `goc`
payload (relative offset) into the second call, then checks that:
  - goc_annotation_raw is traced
  - goc_declared_depends is traced
  - a depends_llm edge appears in GoCMemory

Run:
  python scripts/smoke_test_declared_deps_offline.py
"""

from __future__ import annotations

import io
import json

from src.llm_agent import ToolLoopLLMAgent, ToolLoopConfig
from src.memory import GoCMemory
from src.tools import ToolBox


class DummyLLM:
    def chat(self, *args, **kwargs):
        raise RuntimeError("DummyLLM.chat() should not be called in this smoke test")


def main() -> None:
    mem = GoCMemory()
    cfg = ToolLoopConfig(goc_annotation_mode="hybrid_depends", goc_annotation_gate=False)

    agent = ToolLoopLLMAgent(
        llm=DummyLLM(),
        tools=ToolBox({}),
        mem=mem,
        cfg=cfg,
    )

    buf = io.StringIO()
    agent._trace_fp = buf  # type: ignore

    # Step 0: no goc annotation
    nid1 = agent._record_tool_step(
        step0=0,
        call={"tool": "search", "args": {"query": "x"}},
        tool_name="search",
        args={"query": "x"},
        observation="obs1",
        docids=["D1"],
        task_id="t_smoke",
        method="GoC",
        run_tag="offline",
    )

    # Step 1: declare dependency on previous tool call using relative offset -1
    nid2 = agent._record_tool_step(
        step0=1,
        call={"tool": "open_page", "args": {"docid": "D1"}, "goc": {"d": [-1]}},
        tool_name="open_page",
        args={"docid": "D1"},
        observation="obs2",
        docids=["D1"],
        task_id="t_smoke",
        method="GoC",
        run_tag="offline",
    )

    events = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    types = [e.get("type") for e in events]

    assert "goc_annotation_raw" in types, "Expected goc_annotation_raw in trace"
    assert "goc_declared_depends" in types, "Expected goc_declared_depends in trace"

    # GoCMemory stores adjacency maps by edge type under edges_out.
    out_dep_llm = mem.edges_out.get("depends_llm", {})
    assert nid1 in out_dep_llm and nid2 in out_dep_llm[nid1], "Expected depends_llm edge nid1 -> nid2"

    print("OK: declared deps ingested (", nid1, "->", nid2, ")")

if __name__ == "__main__":
    main()
