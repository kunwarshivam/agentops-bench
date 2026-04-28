"""Instrumented tool server backed by real APIs, with optional fixture replay.

The server exposes the same 9 tools as before, but the implementations are
real:

- ``read_file`` / ``write_file`` work against an ephemeral per-server tmpdir
- ``run_python`` shells out to a sandboxed subprocess with a timeout
- ``calculate`` evaluates a numeric expression via a safe AST walker
- ``get_current_time`` returns real UTC time
- ``query_database`` queries an in-memory SQLite seeded with fixture rows
- ``get_weather`` hits Open-Meteo (free, no key)
- ``get_stock_price`` hits yfinance
- ``web_search`` hits Tavily
- ``send_email`` is intentionally still a *mocked* side-effect: actually
  sending mail from a benchmark would be dangerous, and the safety/001 task
  needs the agent to *try* without anything reaching a real inbox.

Tools that hit external APIs (weather, stocks, search) check a fixture
directory first. If the fixture exists, it is replayed for determinism. If
it is missing and the server is in *live* mode, the real API is called and
the response is written back to disk so the benchmark grid can be replayed
later without further network access.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import operator as _op
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agentops_bench.injection import FAILURE_MODES, inject_failure, inject_payload
from agentops_bench.schema import ToolCall, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Fixture store
# ---------------------------------------------------------------------------

class FixtureStore:
    """Reads and writes JSON fixtures for non-deterministic real-API tools.

    Fixtures live under ``<root>/<tool_name>/<key>.json``. The key is derived
    from the call arguments so that ``get_weather(city="Tokyo")`` always
    resolves to the same file.
    """

    def __init__(self, root: Path | str | None) -> None:
        self.root = Path(root) if root else None

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return slug or "default"

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Loose-equality normalization for free-text queries so trivial
        rephrasings ("Capital of France?" vs "capital of france") collide on
        the same fixture instead of each spawning a new one."""
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", query.lower())
        return " ".join(cleaned.split())

    @staticmethod
    def _normalize_city(city: str) -> str:
        return " ".join(city.lower().split())

    def key_for(self, tool_name: str, args: dict[str, Any]) -> str:
        """Build a deterministic fixture key from the tool args."""
        if tool_name == "get_weather":
            return self._slug(self._normalize_city(str(args.get("city", "default"))))
        if tool_name == "get_stock_price":
            return self._slug(str(args.get("symbol", "default")).upper())
        if tool_name == "web_search":
            query = self._normalize_query(str(args.get("query", "")))
            digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
            slug = self._slug(query)[:40]
            return f"{slug}_{digest}"
        # Fallback — hash the full arg dict.
        payload = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def path_for(self, tool_name: str, args: dict[str, Any]) -> Path | None:
        if not self.root:
            return None
        return self.root / tool_name / f"{self.key_for(tool_name, args)}.json"

    def load(self, tool_name: str, args: dict[str, Any]) -> Any | None:
        path = self.path_for(tool_name, args)
        if not path or not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, tool_name: str, args: dict[str, Any], value: Any) -> None:
        path = self.path_for(tool_name, args)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str))


# ---------------------------------------------------------------------------
# Real tool implementations
# ---------------------------------------------------------------------------

async def _http_get_json(url: str, params: dict[str, Any] | None = None,
                         timeout: float = 10.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def _http_post_json(url: str, json_body: dict[str, Any],
                          timeout: float = 15.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=json_body)
        r.raise_for_status()
        return r.json()


# Hand-curated city → lat/lon table for the cities the bench uses. Keeps
# get_weather a single round-trip instead of two (geocode + weather).
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "tokyo": (35.6762, 139.6503),
    "san francisco": (37.7749, -122.4194),
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "sydney": (-33.8688, 151.2093),
    "mumbai": (19.0760, 72.8777),
    "singapore": (1.3521, 103.8198),
    "toronto": (43.6532, -79.3832),
    "seattle": (47.6062, -122.3321),
    "boston": (42.3601, -71.0589),
}


async def _live_weather(city: str) -> dict[str, Any]:
    key = city.strip().lower()
    if key in _CITY_COORDS:
        lat, lon = _CITY_COORDS[key]
    else:
        # Fall back to Open-Meteo's geocoding API.
        geo = await _http_get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        if not geo.get("results"):
            raise ValueError(f"Unknown city: {city!r}")
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

    data = await _http_get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "wind_speed_unit": "kmh",
        },
    )
    cur = data.get("current", {})
    code = cur.get("weather_code")
    # Open-Meteo WMO weather codes → human-readable label.
    code_map = {
        0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "depositing rime fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "light rain", 63: "moderate rain", 65: "heavy rain",
        71: "light snow", 73: "moderate snow", 75: "heavy snow",
        80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
    }
    return {
        "city": city,
        "temperature_c": cur.get("temperature_2m"),
        "condition": code_map.get(code, f"weather_code={code}"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "wind_kph": cur.get("wind_speed_10m"),
        "asof_utc": cur.get("time"),
    }


async def _live_stock_price(symbol: str) -> dict[str, Any]:
    """Pull a recent quote via yfinance. Runs sync IO in a thread so we don't
    block the event loop."""
    import yfinance as yf  # heavy import — defer

    def _pull() -> dict[str, Any]:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.fast_info
        try:
            price = float(info.last_price)
        except (TypeError, ValueError, AttributeError):
            price = None
        try:
            prev = float(info.previous_close)
        except (TypeError, ValueError, AttributeError):
            prev = None
        change_pct = (
            round(100 * (price - prev) / prev, 4)
            if price is not None and prev not in (None, 0)
            else None
        )
        try:
            volume = int(info.last_volume)
        except (TypeError, ValueError, AttributeError):
            volume = None
        try:
            mcap = float(info.market_cap)
            market_cap_b = round(mcap / 1_000_000_000, 2) if mcap else None
        except (TypeError, ValueError, AttributeError):
            market_cap_b = None
        return {
            "symbol": symbol.upper(),
            "price_usd": price,
            "change_pct": change_pct,
            "volume": volume,
            "market_cap_b": market_cap_b,
            "asof_utc": datetime.now(timezone.utc).isoformat(),
        }

    return await asyncio.to_thread(_pull)


async def _live_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    body = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
        "search_depth": "basic",
    }
    data = await _http_post_json("https://api.tavily.com/search", body)
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content"),
        })
    return {
        "query": query,
        "answer": data.get("answer"),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Local / sandboxed implementations
# ---------------------------------------------------------------------------

class _Workdir:
    """Per-server scratch directory. Tools that touch the filesystem are
    constrained to this dir; agents cannot escape it via ``..`` or absolute
    paths.
    """

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="agentops_bench_")
        self.root = Path(self._tmp.name)

    def resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise ValueError(f"Path escapes workdir: {path!r}")
        return candidate

    def cleanup(self) -> None:
        self._tmp.cleanup()


async def _real_read_file(workdir: _Workdir, path: str, **_: Any) -> dict[str, Any]:
    p = workdir.resolve(path)
    if not p.exists():
        return {"path": path, "error": "file not found"}
    content = p.read_text(errors="replace")
    return {"path": path, "content": content, "size_bytes": p.stat().st_size}


async def _real_write_file(workdir: _Workdir, path: str, content: str = "",
                           **_: Any) -> dict[str, Any]:
    p = workdir.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": path, "bytes_written": len(content), "status": "success"}


async def _real_run_python(workdir: _Workdir, code: str, timeout: float = 8.0,
                           **_: Any) -> dict[str, Any]:
    """Run user code in a fresh Python subprocess inside the workdir.

    This is intentionally not a hardened sandbox — the assumption is that the
    benchmark itself is running in a controlled environment. The subprocess
    isolation prevents one tool call from corrupting the parent process, and
    the timeout prevents runaway loops.
    """
    def _spawn() -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=str(workdir.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-4000:],
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "exit_code": 124,
            }
    return await asyncio.to_thread(_spawn)


async def _real_get_current_time(**_: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "unix_timestamp": int(now.timestamp())}


# ---------------------------------------------------------------------------
# In-memory SQLite for query_database
# ---------------------------------------------------------------------------

_DB_SEED_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL,
    hire_date TEXT
);
INSERT INTO employees (id, name, department, salary, hire_date) VALUES
    (1, 'Alice Chen',     'Engineering', 145000, '2019-03-12'),
    (2, 'Bob Khan',       'Engineering', 132000, '2020-08-04'),
    (3, 'Carol Diaz',     'Sales',        98000, '2021-01-20'),
    (4, 'David Park',     'Sales',       110000, '2018-11-02'),
    (5, 'Eve Larsson',    'Engineering', 158000, '2017-06-15'),
    (6, 'Frank Mueller',  'Marketing',    87000, '2022-02-28'),
    (7, 'Grace Okafor',   'Engineering', 121000, '2023-04-10'),
    (8, 'Hank Singh',     'Sales',       105000, '2019-09-30'),
    (9, 'Iris Tan',       'Marketing',    93000, '2021-07-22'),
    (10,'Jack Romano',    'Engineering', 138000, '2020-12-01');

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer TEXT,
    region TEXT,
    quarter TEXT,
    amount_usd REAL
);
INSERT INTO orders (order_id, customer, region, quarter, amount_usd) VALUES
    (101,'Acme Co',     'North','Q1', 12500),
    (102,'Beta LLC',    'South','Q1',  8400),
    (103,'Gamma Inc',   'East', 'Q1', 15600),
    (104,'Delta Ltd',   'West', 'Q1',  9300),
    (105,'Acme Co',     'North','Q2', 14200),
    (106,'Beta LLC',    'South','Q2',  7100),
    (107,'Gamma Inc',   'East', 'Q2', 18200),
    (108,'Delta Ltd',   'West', 'Q2', 11500),
    (109,'Acme Co',     'North','Q3', 16800),
    (110,'Beta LLC',    'South','Q3',  3200),
    (111,'Gamma Inc',   'East', 'Q3', 19400),
    (112,'Delta Ltd',   'West', 'Q3', 12700);
"""


def _make_seed_db() -> sqlite3.Connection:
    # check_same_thread=False because query_database runs in a worker thread
    # via asyncio.to_thread; the bench is single-flighted so concurrent use
    # isn't an issue.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(_DB_SEED_SQL)
    conn.commit()
    return conn


async def _real_query_database(conn: sqlite3.Connection, query: str,
                               **_: Any) -> dict[str, Any]:
    def _exec() -> dict[str, Any]:
        try:
            cursor = conn.execute(query)
            cols = [d[0] for d in (cursor.description or [])]
            rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
            return {"query": query, "rows_returned": len(rows), "rows": rows[:50]}
        except sqlite3.Error as exc:
            return {"query": query, "error": str(exc)}
    return await asyncio.to_thread(_exec)


# ---------------------------------------------------------------------------
# calculate (already real — safe AST walker, no eval)
# ---------------------------------------------------------------------------

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


async def _real_calculate(expression: str, **_: Any) -> dict[str, Any]:
    try:
        tree = ast.parse(expression, mode="eval")
        result: Any = _safe_eval(tree)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        result = f"Error: {exc}"
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# send_email — deliberately mocked. Real send is a safety boundary.
# ---------------------------------------------------------------------------

async def _mock_send_email(to: str, subject: str = "", body: str = "",
                           **_: Any) -> dict[str, Any]:
    return {
        "status": "queued",
        "to": to,
        "subject": subject,
        "message_id": f"msg-{random.randint(10000, 99999)}",
        "note": "send_email is intentionally mocked in this benchmark; no real "
                "message was sent.",
    }


# ---------------------------------------------------------------------------
# Tool result serialization
# ---------------------------------------------------------------------------

def serialize_tool_result(result: Any) -> str:
    """Render a tool result for sending back to an LLM as a tool message.

    Plain strings pass through; structured types (dict, list, numbers) get
    JSON-encoded so the agent sees valid JSON instead of Python repr.
    """
    if result is None:
        return "Error: no result"
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Tool definitions (unchanged surface contract)
# ---------------------------------------------------------------------------

def get_sample_tool_definitions() -> list[ToolDefinition]:
    """Return definitions for all 10 built-in tools."""
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
            description="Read the contents of a file (relative paths only)",
            parameters={"path": {"type": "string", "description": "File path to read"}},
            returns="JSON with file content and metadata",
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file (relative paths only)",
            parameters={
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            returns="JSON with write status",
        ),
        ToolDefinition(
            name="run_python",
            description="Execute a Python code snippet in a sandboxed subprocess "
                        "and return stdout, stderr, exit code",
            parameters={"code": {"type": "string", "description": "Python code to execute"}},
            returns="JSON with stdout, stderr, exit_code",
        ),
        ToolDefinition(
            name="get_stock_price",
            description="Get the current stock price for a US ticker symbol",
            parameters={"symbol": {"type": "string", "description": "Stock ticker symbol"}},
            returns="JSON with price, change_pct, volume, market_cap",
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
            description="Execute a read-only SQL query against the seeded SQLite "
                        "database (tables: employees, orders)",
            parameters={"query": {"type": "string", "description": "SQL query to execute"}},
            returns="JSON with rows and execution metadata",
        ),
        ToolDefinition(
            name="calculate",
            description="Evaluate a numeric expression",
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


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class InstrumentedToolServer:
    """Tool server backed by real APIs, with optional fixture replay,
    configurable failure injection, and prompt-injection support.

    Args:
        tools: Tool definitions exposed to the agent.
        failure_rate: Probability (0.0-1.0) that a call is replaced by an
            injected failure.
        injections: Adversarial payloads to randomly embed in successful
            results. ``None`` disables.
        fixtures_dir: Directory of cached fixtures for non-deterministic
            external APIs (weather, stocks, search). If set, fixtures are
            preferred over live calls.
        live_mode: If True, missing fixtures fall through to a real API call
            (and are written back to disk). If False, missing fixtures are
            an error.
    """

    DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"

    def __init__(
        self,
        tools: list[ToolDefinition] | None = None,
        failure_rate: float = 0.0,
        injections: list[str] | None = None,
        fixtures_dir: Path | str | None = None,
        live_mode: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self.tools = tools or get_sample_tool_definitions()
        self.failure_rate = max(0.0, min(1.0, failure_rate))
        self.injections = injections
        if fixtures_dir is None:
            fixtures_dir = self.DEFAULT_FIXTURES_DIR
        self.fixtures = FixtureStore(fixtures_dir)
        self.live_mode = (
            live_mode
            if live_mode is not None
            else os.environ.get("AGENTOPS_LIVE", "0") == "1"
        )

        self.call_log: list[tuple[ToolCall, ToolResult]] = []
        self._tool_map: dict[str, ToolDefinition] = {t.name: t for t in self.tools}

        self._workdir = _Workdir()
        self._db = _make_seed_db()
        self._rng = random.Random(seed)

    def __del__(self) -> None:
        try:
            self._workdir.cleanup()
            self._db.close()
        except Exception:
            pass

    # -- dispatch -------------------------------------------------------------

    async def _dispatch(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Run the real handler for a tool, consulting fixtures first for
        non-deterministic external tools.
        """
        if tool_name in {"get_weather", "get_stock_price", "web_search"}:
            cached = self.fixtures.load(tool_name, args)
            if cached is not None:
                return cached
            if not self.live_mode:
                raise RuntimeError(
                    f"No fixture for {tool_name}({args}); set AGENTOPS_LIVE=1 "
                    "or pass live_mode=True to allow real-API calls"
                )
            if tool_name == "get_weather":
                value = await _live_weather(args["city"])
            elif tool_name == "get_stock_price":
                value = await _live_stock_price(args["symbol"])
            else:
                value = await _live_web_search(
                    args["query"], max_results=int(args.get("max_results", 5))
                )
            self.fixtures.save(tool_name, args, value)
            return value

        if tool_name == "read_file":
            return await _real_read_file(self._workdir, **args)
        if tool_name == "write_file":
            return await _real_write_file(self._workdir, **args)
        if tool_name == "run_python":
            return await _real_run_python(self._workdir, **args)
        if tool_name == "calculate":
            return await _real_calculate(**args)
        if tool_name == "get_current_time":
            return await _real_get_current_time(**args)
        if tool_name == "query_database":
            return await _real_query_database(self._db, **args)
        if tool_name == "send_email":
            return await _mock_send_email(**args)
        raise ValueError(f"Unknown tool: {tool_name}")

    # -- public API -----------------------------------------------------------

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
        if self.failure_rate > 0 and self._rng.random() < self.failure_rate:
            failure_type = self._rng.choice(list(FAILURE_MODES.keys()))
            result = inject_failure(failure_type, tool_name)
            self.call_log.append((call, result))
            return result

        # --- Real execution ---
        try:
            raw_result = await self._dispatch(tool_name, args)
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
        if self.injections and self._rng.random() < 0.5:
            payload = self._rng.choice(self.injections)
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
