"""Command line interface.

    studio doctor                       check configuration and what is available
    studio ingest [PATH]                index documents
    studio ask "question"               grounded answer with citations
    studio search "query"               retrieval only, with scores per retriever
    studio agent "task" [--mode ...]    run react | plan | team
    studio eval [--compare]             score the pipeline against a golden set
    studio serve                        start the REST API
    studio ui                           start the Streamlit playground
    studio mcp-serve                    expose studio tools over MCP
    studio mcp-register --config FILE    bridge external MCP tools into agents
    studio tools                        list registered tools
    studio graph                        print the agent graph as Mermaid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_studio import __version__


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str) if not isinstance(payload, str) else payload)


def cmd_doctor(_: argparse.Namespace) -> int:
    from agentic_studio.agents.tools import REGISTRY
    from agentic_studio.llm.router import get_router
    from agentic_studio.rag.embeddings import get_embedder
    from agentic_studio.rag.pipeline import get_pipeline
    from agentic_studio.settings import get_settings

    settings = get_settings()
    optional = {}
    for module in ("sentence_transformers", "faiss", "mcp", "streamlit", "PIL", "llama_cpp"):
        try:
            __import__(module)
            optional[module] = "installed"
        except ImportError:
            optional[module] = "not installed"

    _print(
        {
            "version": __version__,
            "providers": get_router().describe(),
            "embedder": get_embedder().name,
            "corpus": get_pipeline().stats(),
            "tools": REGISTRY.names(),
            "guardrails": {
                "enabled": settings.guardrails.enabled,
                "pii_mode": settings.guardrails.pii_mode,
                "moderation_mode": settings.guardrails.moderation_mode,
            },
            "paths": {"index": str(settings.paths.index), "data_raw": str(settings.paths.data_raw)},
            "optional_packages": optional,
        }
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from agentic_studio.rag.ingest import ingest_directory, ingest_file

    target = Path(args.path) if args.path else None
    if target and target.is_file():
        _print(ingest_file(target))
    else:
        _print(ingest_directory(target))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from agentic_studio.rag.pipeline import get_pipeline

    result = get_pipeline().answer(args.question, top_k=args.top_k)
    if args.json:
        _print(result.to_dict())
        return 0
    print(result.answer)
    print("\nSources:")
    for index, context in enumerate(result.contexts, start=1):
        print(f"  [{index}] {context.chunk.source} (score {context.score:.3f}, {context.retriever})")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from agentic_studio.rag.pipeline import get_pipeline

    contexts, queries = get_pipeline().retrieve(args.query, top_k=args.top_k)
    print(f"Queries searched: {queries}\n")
    for context in contexts:
        print(f"[{context.rank}] {context.chunk.source}  score={context.score:.4f}  via={context.retriever}")
        print(f"    {context.text[:200].replace(chr(10), ' ')}\n")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from agentic_studio.agents.planner import PlanExecuteAgent
    from agentic_studio.agents.react import ToolCallingAgent
    from agentic_studio.agents.supervisor import SupervisorAgent

    if args.mode == "plan":
        agent = PlanExecuteAgent()
    elif args.mode == "team":
        agent = SupervisorAgent()
    else:
        agent = ToolCallingAgent()

    run = agent.run(args.task)
    if args.json:
        _print(run.to_dict())
        return 0

    print(f"Status: {run.status}   steps: {len(run.steps)}   {run.latency_ms:.0f}ms")
    for step in run.steps:
        tools = ", ".join(call.name for call in step.tool_calls)
        print(f"  {step.index}. [{step.node}] {tools or (step.thought[:90] or '-')}")
    if run.pending_approval:
        print("\nPaused for approval:")
        _print(run.pending_approval)
        return 2
    print(f"\n{run.output}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from agentic_studio.evaluation.datasets import default_dataset_path, load_dataset, write_sample_dataset
    from agentic_studio.evaluation.judge import LLMJudge
    from agentic_studio.evaluation.runner import EvalRunner, compare_configs, write_report

    path = Path(args.dataset) if args.dataset else default_dataset_path()
    if not path.exists():
        path = write_sample_dataset(path)
        print(f"Created starter golden set at {path}\n")

    cases = load_dataset(path)
    judge = LLMJudge() if args.judge else None

    if args.compare:
        outcome = compare_configs(cases, judge=judge)
        print("Baseline vs advanced pipeline\n")
        for name, delta in outcome["delta"].items():
            baseline = outcome["baseline"]["aggregate"].get(name, 0.0)
            advanced = outcome["advanced"]["aggregate"].get(name, 0.0)
            print(f"  {name:<20} {baseline:.3f} -> {advanced:.3f}  ({delta:+.3f})")
        return 0

    report = EvalRunner(judge=judge).run(cases, label=args.label)
    print(report.to_markdown())
    paths = write_report(report)
    print(f"Reports: {paths['markdown']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentic_studio.settings import get_settings

    settings = get_settings().api
    uvicorn.run(
        "agentic_studio.api.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app_path = Path(__file__).parent / "ui" / "app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    )


def cmd_mcp_serve(_: argparse.Namespace) -> int:
    from agentic_studio.mcp_bridge.server import main as serve

    serve()
    return 0


def cmd_mcp_register(args: argparse.Namespace) -> int:
    from agentic_studio.mcp_bridge.client import register_from_config_file

    _print(register_from_config_file(Path(args.config)))
    return 0


def cmd_tools(_: argparse.Namespace) -> int:
    from agentic_studio.agents.tools import REGISTRY

    for info in REGISTRY.describe():
        gate = " [approval required]" if info["requires_approval"] else ""
        print(f"{info['name']}{gate}\n    {info['description']}")
    return 0


def cmd_graph(_: argparse.Namespace) -> int:
    from agentic_studio.agents.react import ToolCallingAgent

    print(ToolCallingAgent().app.graph.to_mermaid())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description="AI Agentic Studio")
    parser.add_argument("--version", action="version", version=f"ai-agentic-studio {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check configuration and available components").set_defaults(
        func=cmd_doctor
    )

    ingest = sub.add_parser("ingest", help="index documents")
    ingest.add_argument("path", nargs="?", help="file or directory (default: data/raw)")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="grounded answer with citations")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="retrieval only")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=None)
    search.set_defaults(func=cmd_search)

    agent = sub.add_parser("agent", help="run an agent")
    agent.add_argument("task")
    agent.add_argument("--mode", choices=["react", "plan", "team"], default="react")
    agent.add_argument("--json", action="store_true")
    agent.set_defaults(func=cmd_agent)

    evaluate = sub.add_parser("eval", help="score the pipeline")
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--label", default="golden")
    evaluate.add_argument("--compare", action="store_true", help="compare against a naive baseline")
    evaluate.add_argument("--judge", action="store_true", help="use LLM-as-judge scoring")
    evaluate.set_defaults(func=cmd_eval)

    serve = sub.add_parser("serve", help="start the REST API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", help="start the Streamlit playground")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=cmd_ui)

    sub.add_parser("mcp-serve", help="expose studio tools over MCP").set_defaults(func=cmd_mcp_serve)

    mcp_register = sub.add_parser("mcp-register", help="bridge external MCP tools into agents")
    mcp_register.add_argument("--config", required=True, help="path to an mcpServers JSON file")
    mcp_register.set_defaults(func=cmd_mcp_register)

    sub.add_parser("tools", help="list registered tools").set_defaults(func=cmd_tools)
    sub.add_parser("graph", help="print the agent graph as Mermaid").set_defaults(func=cmd_graph)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
