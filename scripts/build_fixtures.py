"""Pre-build fixture files for the bench's external real-API tools.

Runs each (tool, args) pair once against a live API and saves the response
to ``fixtures/<tool>/<key>.json``. Once these are committed, benchmark runs
are deterministic regardless of network state.

Usage:
    AGENTOPS_LIVE=1 python3 scripts/build_fixtures.py

Pass ``--refresh`` to overwrite existing fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentops_bench.tools import (  # noqa: E402
    FixtureStore,
    InstrumentedToolServer,
)


# Inputs we know the bench tasks (and judge fallbacks) will hit. Adding a
# city / ticker / query that isn't here just means the first run will fall
# through to live mode and write a new fixture.
WEATHER_CITIES = ["Tokyo", "Paris", "London", "San Francisco"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
SEARCH_QUERIES = [
    "what is the capital of france",
    "capital of france",
    "capital of australia",
    "what is the capital of australia",
]


async def snapshot_one(srv: InstrumentedToolServer, store: FixtureStore,
                       tool: str, args: dict, refresh: bool) -> str:
    path = store.path_for(tool, args)
    if path and path.exists() and not refresh:
        return f"  skip {tool} {args} (already cached at {path.relative_to(ROOT)})"
    result = await srv.handle_call(tool, args)
    if result.injected_failure or (
        isinstance(result.result, str) and result.result.startswith("Error:")
    ):
        return f"  FAIL {tool} {args}: {result.result}"
    return f"  ok   {tool} {args} -> {path.relative_to(ROOT) if path else '(no path)'}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Overwrite existing fixtures.")
    ns = parser.parse_args()

    if not os.environ.get("AGENTOPS_LIVE"):
        os.environ["AGENTOPS_LIVE"] = "1"

    fixtures_dir = ROOT / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    srv = InstrumentedToolServer(fixtures_dir=fixtures_dir, live_mode=True)
    store = srv.fixtures

    print(f"Writing fixtures to {fixtures_dir.relative_to(ROOT)}\n")

    print("Weather:")
    for city in WEATHER_CITIES:
        msg = await snapshot_one(srv, store, "get_weather", {"city": city}, ns.refresh)
        print(msg)

    print("\nStocks:")
    for symbol in STOCK_SYMBOLS:
        msg = await snapshot_one(srv, store, "get_stock_price", {"symbol": symbol}, ns.refresh)
        print(msg)

    print("\nSearches:")
    for q in SEARCH_QUERIES:
        msg = await snapshot_one(srv, store, "web_search", {"query": q}, ns.refresh)
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())
