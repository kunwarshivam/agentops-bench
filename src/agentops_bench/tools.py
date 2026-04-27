"""Instrumented tool server with failure injection and call logging."""

from __future__ import annotations

import ast
import asyncio
import json
import operator as _op
import random
import time
from datetime import datetime, timezone
from typing import Any

from agentops_bench.injection import FAILURE_MODES, inject_failure, inject_payload
from agentops_bench.schema import ToolCall, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Mock tool implementations
# ---------------------------------------------------------------------------

async def _web_search(query: str, **_: Any) -> str:
    """Simulate a web search."""
    return json.dumps({
        "query": query,
        "results": [
            {"title": f"Result 1 for '{query}'", "url": "https://example.com/1", "snippet": f"Comprehensive overview of {query}..."},
            {"title": f"Result 2 for '{query}'", "url": "https://example.com/2", "snippet": f"Latest research on {query}..."},
            {"title": f"Result 3 for '{query}'", "url": "https://example.com/3", "snippet": f"Expert analysis of {query}..."},
        ],
    })


async def _get_weather(city: str, **_: Any) -> str:
    """Simulate a weather lookup."""
    weathers = ["sunny", "cloudy", "rainy", "partly cloudy", "snowy"]
    temp = random.randint(-5, 38)
    return json.dumps({
        "city": city,
        "temperature_c": temp,
        "condition": random.choice(weathers),
        "humidity_pct": random.randint(20, 95),
        "wind_kph": random.randint(0, 50),
    })


async def _read_file(path: str, **_: Any) -> str:
    """Simulate reading a file."""
    return json.dumps({
        "path": path,
        "content": f"# Simulated content of {path}\n\nThis is mock file content for benchmarking purposes.\nLine 3 of the file.\nLine 4 of the file.\n",
        "size_bytes": 1234,
    })


async def _write_file(path: str, content: str = "", **_: Any) -> str:
    """Simulate writing a file."""
    return json.dumps({
        "path": path,
        "bytes_written": len(content),
        "status": "success",
    })


async def _run_python(code: str, **_: Any) -> str:
    """Simulate running a Python snippet."""
    return json.dumps({
        "stdout": f"Executed code ({len(code)} chars). Simulated output: OK",
        "stderr": "",
        "exit_code": 0,
    })


async def _get_stock_price(symbol: str, **_: Any) -> str:
    """Simulate fetching a stock price."""
    price = round(random.uniform(50, 500), 2)
    change = round(random.uniform(-5, 5), 2)
    return json.dumps({
        "symbol": symbol.upper(),
        "price_usd": price,
        "change_pct": change,
        "volume": random.randint(1_000_000, 50_000_000),
        "market_cap_b": round(price * random.uniform(1, 20), 1),
    })


async def _send_email(to: str, subject: str = "", body: str = "", **_: Any) -> str:
    """Simulate sending an email."""
    return json.dumps({
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": f"msg-{random.randint(10000, 99999)}",
    })


async def _query_database(query: str, **_: Any) -> str:
    """Simulate a database query."""
    return json.dumps({
        "query": query,
        "rows_returned": random.randint(0, 100),
        "rows": [
            {"id": 1, "name": "Alice", "value": 42.0},
            {"id": 2, "name": "Bob", "value": 37.5},
            {"id": 3, "name": "Charlie", "value": 91.2},
        ],
        "execution_time_ms": random.randint(5, 500),
    })


_MATH_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.USub: _op.neg, ast.UAdd: _op.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


async def _calculate(expression: str, **_: Any) -> str:
    """Evaluate a numeric expression without using eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
        result: Any = _safe_eval(tree)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        result = f"Error: {exc}"
    return json.dumps({"expression": expression, "result": result})


async def _get_current_time(**_: Any) -> str:
    """Return the current UTC time."""
    now = datetime.now(timezone.utc)
    return json.dumps({
        "utc": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
    })


# Registry mapping tool name -> async handler
_MOCK_HANDLERS: dict[str, Any] = {
    "web_search": _web_search,
    "get_weather": _get_weather,
    "read_file": _read_file,
    "write_file": _write_file,
    "run_python": _run_python,
    "get_stock_price": _get_stock_price,
    "send_email": _send_email,
    "query_database": _query_database,
    "calculate": _calculate,
    "get_current_time": _get_current_time,
}


def get_sample_tool_definitions() -> list[ToolDefinition]:
    """Return definitions for all 10 built-in mock tools."""
    return [
        ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters={"query": {"type": "string", "description": "Search query"}},
            returns="JSON with search results",
        ),
        ToolDefinition(
            name="get_weather",
            description="Get current weather for a city",
            parameters={"city": {"type": "string", "description": "City name"}},
            returns="JSON with temperature, condition, humidity, wind",
        ),
        ToolDefinition(
            name="read_file",
            description="Read the contents of a file",
            parameters={"path": {"type": "string", "description": "File path to read"}},
            returns="JSON with file content and metadata",
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file",
            parameters={
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            returns="JSON with write status",
        ),
        ToolDefinition(
            name="run_python",
            description="Execute a Python code snippet and return the output",
            parameters={"code": {"type": "string", "description": "Python code to execute"}},
            returns="JSON with stdout, stderr, exit_code",
        ),
        ToolDefinition(
            name="get_stock_price",
            description="Get the current stock price for a ticker symbol",
            parameters={"symbol": {"type": "string", "description": "Stock ticker symbol"}},
            returns="JSON with price, change, volume",
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email message",
            parameters={
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            returns="JSON with send status and message ID",
        ),
        ToolDefinition(
            name="query_database",
            description="Execute a SQL query against the database",
            parameters={"query": {"type": "string", "description": "SQL query to execute"}},
            returns="JSON with rows and execution time",
        ),
        ToolDefinition(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
            returns="JSON with the computed result",
        ),
        ToolDefinition(
            name="get_current_time",
            description="Get the current UTC date and time",
            parameters={},
            returns="JSON with UTC time and unix timestamp",
        ),
    ]


class InstrumentedToolServer:
    """Tool server that wraps mock tools with failure and injection capabilities.

    Args:
        tools: Tool definitions available to the agent.
        failure_rate: Probability (0.0-1.0) that any given call will be
            replaced with a random failure.
        injections: List of adversarial payloads to randomly embed in
            successful tool results. Pass ``None`` to disable.
    """

    def __init__(
        self,
        tools: list[ToolDefinition] | None = None,
        failure_rate: float = 0.0,
        injections: list[str] | None = None,
    ) -> None:
        self.tools = tools or get_sample_tool_definitions()
        self.failure_rate = max(0.0, min(1.0, failure_rate))
        self.injections = injections
        self.call_log: list[tuple[ToolCall, ToolResult]] = []
        self._tool_map: dict[str, ToolDefinition] = {t.name: t for t in self.tools}

    async def handle_call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool call, potentially injecting failures or payloads.

        Args:
            tool_name: Name of the tool to invoke.
            args: Arguments to pass to the tool.

        Returns:
            A ToolResult, possibly with injected failures or payloads.
        """
        call = ToolCall(
            tool_name=tool_name,
            arguments=args,
            timestamp=datetime.now(timezone.utc),
        )

        start = time.monotonic()

        # --- Failure injection ---
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            failure_type = random.choice(list(FAILURE_MODES.keys()))
            result = inject_failure(failure_type, tool_name)
            self.call_log.append((call, result))
            return result

        # --- Normal execution ---
        handler = _MOCK_HANDLERS.get(tool_name)
        if handler is None:
            result = ToolResult(
                tool_name=tool_name,
                result=f"Error: unknown tool '{tool_name}'",
                latency_ms=0.0,
            )
            self.call_log.append((call, result))
            return result

        try:
            raw_result = await handler(**args)
        except Exception as exc:
            result = ToolResult(
                tool_name=tool_name,
                result=f"Error: {exc}",
                latency_ms=(time.monotonic() - start) * 1000,
            )
            self.call_log.append((call, result))
            return result

        elapsed_ms = (time.monotonic() - start) * 1000

        # --- Prompt injection ---
        if self.injections and random.random() < 0.5:
            payload = random.choice(self.injections)
            result = inject_payload(raw_result, payload)
            result.tool_name = tool_name
            result.latency_ms = elapsed_ms
            self.call_log.append((call, result))
            return result

        result = ToolResult(
            tool_name=tool_name,
            result=raw_result,
            latency_ms=elapsed_ms,
        )
        self.call_log.append((call, result))
        return result

    def get_tool_definitions_for_agent(self) -> list[dict[str, Any]]:
        """Return tool definitions in a format suitable for LLM function-calling APIs."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": t.parameters,
                },
            }
            for t in self.tools
        ]

    def reset_log(self) -> None:
        """Clear the call log."""
        self.call_log.clear()
