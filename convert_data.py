"""Convert the real EW5d account ledger into website JSON/JavaScript assets.

This module never recomputes strategy returns from prediction rows.  Positions,
NAV and friction all come from the executable account ledger produced by
``live_portfolio.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_START = pd.Timestamp("2026-07-20")
INITIAL_CAPITAL = 200_000.0
_STOCK_NAME_CACHE: dict[str, str] | None = None


def _stock_names() -> dict[str, str]:
    global _STOCK_NAME_CACHE
    if _STOCK_NAME_CACHE is not None:
        return _STOCK_NAME_CACHE
    co_path = Path("dataset/input/original_data/TRD_Co.csv")
    if not co_path.exists():
        _STOCK_NAME_CACHE = {}
        return _STOCK_NAME_CACHE
    raw = pd.read_csv(co_path, dtype={"Stkcd": str})
    raw["Stkcd"] = raw["Stkcd"].str.strip().str.zfill(6)
    _STOCK_NAME_CACHE = dict(zip(raw["Stkcd"], raw["Stknme"]))
    return _STOCK_NAME_CACHE

REGIME_MAP = {
    "Bull+LowVol":  {"label": "低波动牛市", "advice": "趋势稳健，波动可控。维持仓位，逢回调可加仓。"},
    "Bull+HighVol": {"label": "高波动牛市", "advice": "趋势向上但波动加剧，警惕赶顶。考虑分批止盈，适度降低弹性。"},
    "Bear+LowVol":  {"label": "低波动熊市", "advice": "阴跌格局，控制仓位。耐心等待右侧信号，不宜急于抄底。"},
    "Bear+HighVol": {"label": "高波动熊市", "advice": "波动剧烈且趋势向下。收紧仓位至最低，空仓睡觉战法。"},
    "Unknown":      {"label": "数据预热中", "advice": "样本不足，暂无判断。"},
}
INCLUDE_FIELDS = [
    "date", "market_ret_1d", "market_nav", "regime_score", "bull_score",
    "bull", "state_age", "switch_flag", "vol_percentile", "vol_score",
    "high_vol", "regime", "drawdown", "trend_score", "breadth_score",
    "breadth_up_1d_raw", "breadth_pos_20d_raw", "breadth_pos_60d_raw",
    "trend_20_score", "trend_60_score", "trend_120_score",
]


def _value(value):
    if value == "" or value is None:
        return None
    try:
        number = float(value)
        return None if not np.isfinite(number) else number
    except (ValueError, TypeError):
        return value


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace([np.inf, -np.inf], np.nan).copy()
    for col in clean.select_dtypes(include=["datetime", "datetimetz"]).columns:
        clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def build_market_data(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = {field: _value(row[field]) for field in INCLUDE_FIELDS if field in row}
            regime = out.get("regime", "Unknown")
            out["regime_cn"] = REGIME_MAP.get(regime, {}).get("label", regime)
            out["advice"] = REGIME_MAP.get(regime, {}).get("advice", "")
            rows.append(out)
    latest = dict(rows[-1]) if rows else {}
    for lookback in (60, 20, 5, 0):
        index = len(rows) - 1 - lookback
        anchor = rows[index] if index >= 0 else None
        for key in ("regime_score", "trend_score", "breadth_score"):
            latest[f"{key}_{lookback}d" if lookback else key] = anchor.get(key) if anchor else None
    return {"latest": latest, "data": rows}


def _published_nav(account: pd.DataFrame, start: pd.Timestamp, capital: float) -> pd.DataFrame:
    """Public series has an explicit inception anchor NAV=1 at 7/20 close.

    Entry-day orders are visible in holdings/friction.  Their costs enter the
    first subsequent realised NAV, so the start point remains exactly one.
    """
    work = account[account["date"] >= start].copy().sort_values("date")
    if work.empty:
        raise ValueError(f"Account ledger has no rows on/after {start.date()}")
    work["published_nav"] = work["equity"] / capital
    if (work["date"] == start).any():
        work.loc[work["date"] == start, "published_nav"] = 1.0
    else:
        anchor = {c: np.nan for c in work.columns}
        anchor.update({"date": start, "equity": capital, "cash": capital,
                       "market_value": 0.0, "n_positions": 0, "published_nav": 1.0})
        work = pd.concat([pd.DataFrame([anchor]), work], ignore_index=True).sort_values("date")
    work["published_return"] = work["published_nav"].pct_change().fillna(0.0)
    work["cum_return"] = work["published_nav"] - 1.0
    peak = work["published_nav"].cummax()
    work["drawdown"] = work["published_nav"] / peak - 1.0
    return work


def _compute_benchmark(prices: pd.DataFrame, dates: pd.Series, start: pd.Timestamp,
                       universe: pd.DataFrame | None = None) -> pd.DataFrame:
    prices["time"] = pd.to_datetime(prices["time"])
    prices[["close", "pre_close"]] = prices[["close", "pre_close"]].apply(pd.to_numeric, errors="coerce")
    prices["ret"] = prices["close"] / prices["pre_close"] - 1.0
    prices = prices.replace([np.inf, -np.inf], np.nan)
    if universe is not None and not universe.empty:
        keys = universe[["time", "stock_id"]].drop_duplicates().copy()
        keys["time"] = pd.to_datetime(keys["time"])
        keys["stock_id"] = keys["stock_id"].astype(str).str.strip().str.zfill(6)
        prices["stock_id"] = prices["stock_id"].astype(str).str.strip().str.zfill(6)
        prices = prices.merge(keys, on=["time", "stock_id"], how="inner")
    daily = prices[prices["time"] > start].groupby("time")["ret"].mean().sort_index()
    wanted = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    out = pd.DataFrame(index=wanted)
    out["return"] = daily.reindex(wanted).fillna(0.0)
    out["nav"] = (1.0 + out["return"]).cumprod()
    if start in out.index:
        out.loc[start, ["return", "nav"]] = [0.0, 1.0]
        later = out.index > start
        out.loc[later, "nav"] = (1.0 + out.loc[later, "return"]).cumprod()
    out.index.name = "date"
    return out.reset_index()


def _zz1500_baseline(dates: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """ZZ1500 = ZZ500 + ZZ1000 official index, free-float market-cap weighted (≈6:5)."""
    import pandas as pd
    zz500 = pd.read_parquet("dataset/input/zz500_daily.parquet")
    zz1000 = pd.read_parquet("dataset/input/zz1000_daily.parquet")
    zz500 = zz500.set_index("time")["pct_chg"]
    zz1000 = zz1000.set_index("time")["pct_chg"]
    ret = (zz500 * 96735.73 + zz1000 * 80015.19) / (96735.73 + 80015.19) / 100.0
    daily = ret[ret.index > start].sort_index()
    wanted = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    out = pd.DataFrame(index=wanted)
    out["return"] = daily.reindex(wanted).fillna(0.0)
    out["nav"] = (1.0 + out["return"]).cumprod()
    if start in out.index:
        out.loc[start, ["return", "nav"]] = [0.0, 1.0]
        out.loc[out.index > start, "nav"] = (1.0 + out.loc[out.index > start, "return"]).cumprod()
    out.index.name = "date"
    return out.reset_index()


def _today_holdings(snapshots: pd.DataFrame, latest: pd.Timestamp) -> list[dict]:
    day = snapshots[snapshots["date"] == latest].copy()
    if day.empty:
        return []
    day = day.sort_values(["weight", "score"], ascending=False)
    names = _stock_names()
    day["stock_name"] = day["stock_id"].map(names)
    keep = ["stock_id", "stock_name", "shares", "lots", "avg_cost", "last_price", "market_value",
            "weight", "unrealized_pnl", "score", "rank_pct", "entry_date",
            "last_buy_date", "selected_today", "delayed_exit_days"]
    return _records(day[[c for c in keep if c in day.columns]])


def _yesterday_performance(
    snapshots: pd.DataFrame, trades: pd.DataFrame, latest: pd.Timestamp,
) -> tuple[str | None, list[dict]]:
    prior_dates = sorted(d for d in snapshots["date"].dropna().unique() if pd.Timestamp(d) < latest)
    if not prior_dates:
        return None, []
    previous = pd.Timestamp(prior_dates[-1])
    prev = snapshots[snapshots["date"] == previous].copy()
    current = snapshots[snapshots["date"] == latest][["stock_id", "last_price", "shares"]].copy()
    current = current.rename(columns={"last_price": "today_price", "shares": "today_shares"})
    sells = trades[(trades["date"] == latest) & (trades["action"] == "SELL")].copy()
    sell_px = sells.groupby("stock_id")["price"].mean().rename("sell_price") if not sells.empty else pd.Series(name="sell_price", dtype=float, index=pd.Index([], name="stock_id"))
    sell_fee = sells.groupby("stock_id")["fee"].sum().rename("sell_fee") if not sells.empty else pd.Series(name="sell_fee", dtype=float, index=pd.Index([], name="stock_id"))
    work = prev.merge(current, on="stock_id", how="left").merge(sell_px, on="stock_id", how="left").merge(sell_fee, on="stock_id", how="left")
    work["valuation_price"] = work["today_price"].fillna(work["sell_price"]).fillna(work["last_price"])
    work["return_pct"] = (work["valuation_price"] / work["last_price"] - 1.0) * 100.0
    work["gross_pnl"] = work["shares"] * (work["valuation_price"] - work["last_price"])
    work["sell_fee"] = work["sell_fee"].fillna(0.0)
    work["net_pnl"] = work["gross_pnl"] - work["sell_fee"]
    work["status"] = np.where(work["today_shares"].fillna(0) <= 0, "sold",
                              np.where(work["today_shares"] < work["shares"], "reduced", "held"))
    names = _stock_names()
    work["stock_name"] = work["stock_id"].map(names)
    keep = ["stock_id", "stock_name", "shares", "lots", "last_price", "valuation_price", "return_pct",
            "gross_pnl", "sell_fee", "net_pnl", "status"]
    return previous.strftime("%Y-%m-%d"), _records(work[keep].sort_values("return_pct", ascending=False))


def build_portfolio_data(evidence_dir: Path, price_path: Path,
                         start: pd.Timestamp = TRACK_START,
                         capital: float = INITIAL_CAPITAL) -> dict:
    account = pd.read_parquet(evidence_dir / "real_account_ew5d.parquet")
    snapshots = pd.read_parquet(evidence_dir / "real_positions_ew5d.parquet")
    trades = pd.read_parquet(evidence_dir / "real_trades_ew5d.parquet")
    orders_path = evidence_dir / "real_orders_ew5d.parquet"
    orders = pd.read_parquet(orders_path) if orders_path.exists() else pd.DataFrame()
    scores_path = evidence_dir / "real_scores_ew5d.parquet"
    score_universe = pd.read_parquet(scores_path, columns=["time", "stock_id"]) if scores_path.exists() else None
    summary_path = evidence_dir / "real_summary_ew5d.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for frame, columns in ((account, ["date"]), (snapshots, ["date", "entry_date", "last_buy_date"]),
                           (trades, ["date", "entry_date", "signal_date"]), (orders, ["date"])):
        for col in columns:
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col])

    public = _published_nav(account, start, capital)
    benchmark = _zz1500_baseline(public["date"], start)
    joined = public.merge(benchmark.rename(columns={"nav": "benchmark_nav", "return": "benchmark_return"}),
                          on="date", how="left")
    joined["excess_nav"] = joined["published_nav"] / joined["benchmark_nav"]
    latest_date = pd.Timestamp(joined["date"].max())
    today = _today_holdings(snapshots, latest_date)
    previous_date, yesterday = _yesterday_performance(snapshots, trades, latest_date)

    friction_cols = [c for c in ("commission", "transfer_fee", "stamp_tax", "slippage_cost", "friction_cost") if c in joined]
    friction = {f"total_{c}": round(float(joined[c].fillna(0).sum()), 2) for c in friction_cols}
    friction["daily"] = _records(joined[["date", *friction_cols]].fillna(0))
    friction["assumptions"] = summary.get("execution_assumptions", {
        "commission_rate_each_side": 0.0003,
        "minimum_commission_per_order": 0.0,
        "stamp_tax_sell_rate": 0.0005,
        "transfer_fee_each_side_rate": 0.0,
        "buy_slippage_bps_by_tier": {"1": 3.0, "2": 5.0, "3": 8.0},
        "sell_slippage_bps": 3.0,
    })

    nav_history = pd.DataFrame({
        "date": joined["date"], "nav": joined["published_nav"],
        "ret_pct": joined["published_return"] * 100.0,
        "cum_ret_pct": joined["cum_return"] * 100.0,
        "drawdown_pct": joined["drawdown"] * 100.0,
        "zz1500_nav": joined["benchmark_nav"],
        "zz1500_ret_pct": joined["benchmark_return"] * 100.0,
        "excess_nav": joined["excess_nav"],
    })
    daily_cols = ["date", "equity", "cash", "market_value", "n_positions", "n_buys", "n_sells",
                  "blocked_buys", "blocked_sells", "turnover", "fees", "slippage_cost",
                  "friction_cost", "realized_pnl", "unrealized_pnl"]
    daily = joined[[c for c in daily_cols if c in joined.columns]].copy()
    daily = daily.merge(nav_history[["date", "nav", "ret_pct", "cum_ret_pct"]], on="date", how="left")

    last = joined.iloc[-1]
    blocked = orders[(orders.get("status", pd.Series(dtype=str)).isin(["BLOCKED", "BLOCKED_T1", "REJECTED"]))]
    latest_blocked = blocked[blocked["date"] == latest_date] if not blocked.empty else blocked
    return {
        "track_start": start.strftime("%Y-%m-%d"),
        "strategy": "EW Rank 10d Top50",
        "latest": {
            "date": latest_date.strftime("%Y-%m-%d"), "nav": round(float(last["published_nav"]), 6),
            "daily_return_pct": round(float(last["published_return"]) * 100, 4),
            "cum_return_pct": round(float(last["cum_return"]) * 100, 4),
            "equity": round(float(last.get("equity", capital)), 2),
            "cash": round(float(last.get("cash", capital)), 2),
            "invested_value": round(float(last.get("market_value", 0)), 2),
            "invested_pct": round(float(last.get("market_value", 0)) / float(last.get("equity", capital)) * 100, 2),
            "n_positions": len(today), "initial_capital": capital,
            "zz1500_nav": round(float(last["benchmark_nav"]), 6),
            "excess_nav": round(float(last["excess_nav"]), 6),
        },
        "today_holdings": today,
        "yesterday": {"date": previous_date, "holdings": yesterday},
        "nav_history": _records(nav_history),
        "daily_summary": _records(daily),
        "friction": friction,
        "blocked_orders_today": _records(latest_blocked) if not latest_blocked.empty else [],
        "definitions": {
            "nav_anchor": "2026-07-20 close, NAV=1.0 before inception orders",
            "today_holdings": "actual post-trade holdings; return is not yet shown",
            "yesterday_holdings": "previous close holdings and their latest close-to-close contribution",
            "benchmark": "ZZ500+ZZ1000 official index (free-float market-cap weighted)",
        },
    }


def _write_js_json(value: dict, js_path: Path, variable: str, json_path: Path | None = None) -> None:
    js_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tmp = js_path.with_suffix(js_path.suffix + ".tmp")
    tmp.write_text(f"// Auto-generated\nwindow.{variable} = {payload};\n", encoding="utf-8")
    tmp.replace(js_path)
    if json_path is not None:
        jtmp = json_path.with_suffix(json_path.suffix + ".tmp")
        jtmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        jtmp.replace(json_path)


def main() -> None:
    root = Path(os.environ.get("JULONG_ROOT", r"D:\JulongQuant"))
    parser = argparse.ArgumentParser(description="Generate website data from the real account ledger")
    parser.add_argument("--market-csv", type=Path, default=root / "dataset/input/market_regime.csv")
    parser.add_argument("--evidence-dir", type=Path, default=root / "reports/strategy_v1/evidence")
    parser.add_argument("--price-path", type=Path, default=root / "dataset/processed/unified_daily_panel.parquet")
    parser.add_argument("--web-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start", default="2026-07-20")
    parser.add_argument("--capital", type=float, default=200_000.0)
    parser.add_argument("--dual", action="store_true",
                        help="Also generate portfolio_s2.js from defend strategy")
    parser.add_argument("--triple", action="store_true",
                        help="Generate all three: baseline + defend + elite")
    args = parser.parse_args()

    market = build_market_data(args.market_csv)
    portfolio = build_portfolio_data(args.evidence_dir, args.price_path,
                                     pd.Timestamp(args.start), args.capital)
    _write_js_json(market, args.web_dir / "data.js", "MARKET_DATA",
                   args.web_dir / "market_data.json")
    _write_js_json(portfolio, args.web_dir / "portfolio.js", "PORTFOLIO_DATA",
                   args.web_dir / "portfolio_data.json")
    latest = portfolio["latest"]
    print(f"基准策略 through {latest['date']}: NAV={latest['nav']:.6f}, "
          f"equity={latest['equity']:,.2f}, positions={latest['n_positions']}")

    if args.dual or args.triple:
        s2_dir = args.evidence_dir.parent / "evidence_s2"
        portfolio_s2 = build_portfolio_data(s2_dir, args.price_path,
                                            pd.Timestamp(args.start), args.capital)
        _write_js_json(portfolio_s2, args.web_dir / "portfolio_s2.js",
                       "PORTFOLIO_DATA_S2",
                       args.web_dir / "portfolio_s2_data.json")
        latest_s2 = portfolio_s2["latest"]
        print(f"防守策略 through {latest_s2['date']}: NAV={latest_s2['nav']:.6f}, "
              f"equity={latest_s2['equity']:,.2f}, positions={latest_s2['n_positions']}")

    if args.triple:
        s3_dir = args.evidence_dir.parent / "evidence_s3"
        portfolio_s3 = build_portfolio_data(s3_dir, args.price_path,
                                            pd.Timestamp(args.start), args.capital)
        _write_js_json(portfolio_s3, args.web_dir / "portfolio_s3.js",
                       "PORTFOLIO_DATA_S3",
                       args.web_dir / "portfolio_s3_data.json")
        latest_s3 = portfolio_s3["latest"]
        print(f"集中策略 through {latest_s3['date']}: NAV={latest_s3['nav']:.6f}, "
              f"equity={latest_s3['equity']:,.2f}, positions={latest_s3['n_positions']}")


if __name__ == "__main__":
    main()
