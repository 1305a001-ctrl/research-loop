# research-loop

> Continuous research loop — Sharpe tracker + auto-halt.

Periodically reads per-strategy aggregated stats from Postgres (`positions`
joined with `strategies`), computes a Sharpe proxy + halt decision, and
optionally actuates halts via Redis `strategy:halts`. Mirrors the
control-plane dashboard's StrategyFitnessGrid logic so the two views
stay aligned.

## Why

`risk-watcher` already handles drawdown + correlation alerts; that's
position-level. `research-loop` is *strategy-level*: catches strategies
that are statistically bleeding even when individual positions look fine.

It implements the Phase 7 commitment from the Chainlink Hybrid plan:
> "Continuous research loop — nightly backtest + auto-Sharpe tracker +
> auto-halt below Sharpe 0"

## Architecture

```
Postgres (positions, strategies)
        │
        ▼
research-loop (Python, ai-edge)
        │
        ├─→ sharpe_proxy(per-trade pnl) → StrategyStats
        ├─→ should_halt(stats) → HaltDecision
        └─→ Redis SADD strategy:halts <slug>  (gated by ENFORCE_HALTS)
                │
                ▼
        strategy-runners + liquidation-bot read this SET
        and stop firing the halted strategy on next cycle
```

## Halt criteria (defaults, conservative)

A strategy is auto-halted when **all** of:
- `n_closed >= 30` trades in the measurement window (default 7 days), OR
- `total_pnl_usd <= -$200` (catastrophic loss — overrides sample-size rule)

AND any of:
- `sharpe < 0` (losing money on average), OR
- `total_pnl_usd <= catastrophic_pnl_floor_usd`

Sample-size + day-count thresholds prevent false halts on small samples.
Catastrophic-loss path bypasses sample-size so a single $500 disaster
trade triggers an immediate halt.

## Enforcement

`ENFORCE_HALTS=false` (default) → logs what it WOULD halt but doesn't
write. Useful for the first week of observation.

`ENFORCE_HALTS=true` → actually adds to `strategy:halts`. Strategies
unhalt only manually (via `SREM strategy:halts <slug>`).

## Run

### Local dev
```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

### ai-edge deployment
```bash
# 1. /srv/secrets/research-loop.env on ai-edge — copy from .env.example,
#    fill REDIS_URL (must be reachable from ai-edge) + POSTGRES_DSN
# 2. Deploy:
ssh ai-edge 'sudo docker compose -f /srv/compose/research-loop/docker-compose.yml up -d'
# 3. Verify:
ssh ai-edge 'curl -s localhost:8015/health' | python3 -m json.tool
```

If ai-edge can't reach ai-primary's Redis/Postgres directly, run on
ai-primary instead (same compose, same env).

## Verifying it's working

```bash
# tail logs — should see one cycle per hour
ssh <host> 'sudo docker logs --tail 60 research-loop'

# look for lines like:
#   research_loop.cycle n_evaluated=22 n_halted=2 n_would_halt=0
#   research_loop.strategy arb-momentum-1m@binance n=13 pnl=$-110.97 sharpe=-0.42 [bleed] HALT
```

## What's deferred (next session)

- **Nightly backtest engine** that re-validates strategies against
  historical data (uses Chainlink Candlestick API — same creds as
  Data Streams)
- **Telegram report** to pa-agent bot summarizing daily halts + new
  strategy promotions
- **Per-strategy fitness Redis publish** (`strategy:fitness:<slug>`) so
  control-plane reads from there instead of computing per-request

## Tests

```bash
PYTHONPATH=src pytest -q
```

19 tests, all pure-math, no I/O.

## License

MIT.
