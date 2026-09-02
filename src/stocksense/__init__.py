"""StockSense — a local quant stack for NSE cash equities.

Layering rule (enforced by tests/unit/test_layering.py): a module may import
from layers ABOVE it in this list, never below.

    core          config, IST calendar, typed domain objects, clock
    data          ingestion + the DuckDB/Parquet store
    microstructure  order book, spread, impact, order flow
    features      factor registry and feature construction
    labels        forward returns, path-dependent first-touch labels
    strategies    parameterised strategy families
    search        the generator and the selection funnel
    simulation    Monte Carlo, risk, position sizing
    evaluation    metrics, robustness, walk-forward, gate, vault, attempts
    execution     costs, fills, algos, paper broker, interlocks
    brokers       live broker adapters (the only place orders may be placed)
    live          the intraday engine
    portfolio     forensics on the user's real trades
    llm           Ollama, Claude CLI, Obsidian knowledge graph
    server        FastAPI + WebSocket
    cli           command entry points
"""

__version__ = "3.0.0"
