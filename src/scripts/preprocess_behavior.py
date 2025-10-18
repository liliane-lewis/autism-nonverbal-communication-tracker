#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess behavioral, sleep, and goal data for ML.

What this script does
---------------------
1) Connects to PostgreSQL (SQLAlchemy) using env vars or CLI args.
2) Builds CHILD-DATE aggregates from raw behavior_entry (robust even if
   views are missing or different).
3) Joins with sleep_log (per child per date) and goal status (per child).
4) Engineers features:
   - attempts per behavior type (eye_contact, gesture, vocalization, imitation, etc.)
   - total_events
   - avg_prompt_level and pct_independent
   - communication_score (weighted attempts)
   - sleep_deficit (max(0, 8 - sleep_hours))
   - 7-day rolling means per child for stability
5) Outputs tidy data to CSV and Parquet in ./data

Dependencies
------------
pip install pandas sqlalchemy psycopg2-binary pyarrow

Usage
-----
python3 preprocess_behavior.py \
  --db-host localhost --db-port 5432 --db-name postgres --db-user postgres --db-pass 1234 \
  --out-dir ./data

Environment variables (optional, override by CLI)
-------------------------------------------------
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# ----------------------------
# CLI and configuration
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Autism Tracker data for ML")
    p.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    p.add_argument("--db-name", default=os.getenv("DB_NAME", "postgres"))
    p.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    p.add_argument("--db-pass", default=os.getenv("DB_PASS", "1234"))
    p.add_argument("--out-dir", default="./data", help="Directory to write outputs")
    p.add_argument("--rolling", type=int, default=7, help="Rolling window (days)")
    return p.parse_args()


# ----------------------------
# SQL helpers
# ----------------------------
SQL_BEHAVIOR_CHILD_DAILY = """
/*
Build child-date aggregates directly from behavior_entry + behavior_type.
This avoids depending on views that may not include child_id.
*/
WITH base AS (
  SELECT
    be.child_id,
    (be.ts AT TIME ZONE 'UTC')::date AS day_utc,
    bt.name AS behavior_type,
    be.prompt_level
  FROM behavior_entry be
  JOIN behavior_type bt ON bt.id = be.behavior_type_id
)
SELECT
  child_id,
  day_utc,
  COUNT(*)::int                                   AS total_events,
  AVG(prompt_level)::float                         AS avg_prompt_level,
  100.0 * AVG(CASE WHEN prompt_level <= 1 THEN 1.0 ELSE 0.0 END) AS pct_independent,
  COUNT(*) FILTER (WHERE behavior_type = 'eye_contact')::int     AS eye_contact,
  COUNT(*) FILTER (WHERE behavior_type = 'gesture')::int         AS gesture,
  COUNT(*) FILTER (WHERE behavior_type = 'vocalization')::int    AS vocalization,
  COUNT(*) FILTER (WHERE behavior_type = 'imitation')::int       AS imitation,
  COUNT(*) FILTER (WHERE behavior_type = 'joint_attention')::int AS joint_attention,
  COUNT(*) FILTER (WHERE behavior_type = 'self_regulation')::int AS self_regulation
FROM base
GROUP BY child_id, day_utc
ORDER BY child_id, day_utc;
"""

SQL_SLEEP = """
SELECT
  child_id,
  "date"::date AS day_utc,
  sleep_hours::float,
  sleep_quality
FROM sleep_log
ORDER BY child_id, day_utc;
"""

SQL_GOAL_STATUS = """
/*
Goal status per child. The schema we shipped exposes columns:
child_id, achieved, ongoing, not_achieved.
If you applied the compatibility patch, there are also legacy aliases.
*/
SELECT
  child_id,
  COALESCE(achieved, 0)::int       AS goals_achieved,
  COALESCE(ongoing, 0)::int        AS goals_ongoing,
  COALESCE(not_achieved, 0)::int   AS goals_not_achieved
FROM v_goal_status
ORDER BY child_id;
"""

SQL_CHILD = "SELECT id AS child_id, code AS child_code FROM child ORDER BY id;"


# ----------------------------
# Feature engineering helpers
# ----------------------------
def engineering(df: pd.DataFrame, rolling: int) -> pd.DataFrame:
    """
    Add engineered features and rolling statistics per child.
    """
    # Basic derived features
    df["sleep_deficit"] = np.clip(8.0 - df["sleep_hours"], a_min=0.0, a_max=None)

    # Simple weighted communication score (tune weights as needed)
    # Emphasize spontaneous/functional communication attempts
    df["communication_score"] = (
        1.2 * df["eye_contact"]
        + 1.5 * df["gesture"]
        + 1.4 * df["vocalization"]
        + 1.6 * df["imitation"]
        + 1.0 * df["joint_attention"]
    )

    # Percent-based features in [0, 100] already; guard against NaNs
    for col in ["pct_independent"]:
        df[col] = df[col].fillna(0.0)

    # Rolling stats per child to smooth daily noise
    df = df.sort_values(["child_id", "day_utc"])
    group = df.groupby("child_id", group_keys=False)

    roll_cols_mean = [
        "total_events",
        "avg_prompt_level",
        "pct_independent",
        "sleep_hours",
        "sleep_deficit",
        "communication_score",
        "eye_contact",
        "gesture",
        "vocalization",
        "imitation",
        "joint_attention",
        "self_regulation",
    ]

    for col in roll_cols_mean:
        df[f"{col}_roll{rolling}"] = group[col].apply(lambda x: x.rolling(rolling, min_periods=1).mean())

    # Example target-like helper for early ML experiments:
    # "progress_label" = 1 if pct_independent improved vs 7-day mean by a margin
    df["progress_label"] = (
        (df["pct_independent"] >= (df["pct_independent_roll7"] + 5.0)).astype(int)
        if f"pct_independent_roll{rolling}" in df.columns
        else 0
    )

    return df


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build SQLAlchemy connection string
    url = f"postgresql+psycopg2://{args.db_user}:{args.db_pass}@{args.db_host}:{args.db_port}/{args.db_name}"
    engine = create_engine(url)

    # Load child-date behavior aggregates
    behavior = pd.read_sql(text(SQL_BEHAVIOR_CHILD_DAILY), engine)
    if behavior.empty:
        raise SystemExit("No behavior data found. Seed data before preprocessing.")

    # Load sleep
    sleep = pd.read_sql(text(SQL_SLEEP), engine)

    # Load goal status per child
    # If the view does not exist, skip gracefully with zeros
    try:
        goal_status = pd.read_sql(text(SQL_GOAL_STATUS), engine)
    except Exception:
        goal_status = pd.DataFrame(columns=["child_id", "goals_achieved", "goals_ongoing", "goals_not_achieved"])

    # Load child codes for readability
    child = pd.read_sql(text(SQL_CHILD), engine)

    # Merge behavior + sleep on child_id, day_utc (left join keeps behavior days even if sleep missing)
    df = behavior.merge(sleep, how="left", on=["child_id", "day_utc"])

    # Merge goal status (per child, not daily)
    df = df.merge(goal_status, how="left", on="child_id")
    df[["goals_achieved", "goals_ongoing", "goals_not_achieved"]] = df[
        ["goals_achieved", "goals_ongoing", "goals_not_achieved"]
    ].fillna(0).astype(int)

    # Add child codes
    df = df.merge(child, how="left", on="child_id")

    # Type and missing-value handling
    df["day_utc"] = pd.to_datetime(df["day_utc"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Feature engineering and rolling stats
    df = engineering(df, rolling=args.rolling)

    # Sort and write outputs
    df = df.sort_values(["child_id", "day_utc"]).reset_index(drop=True)

    csv_path = out_dir / "behavior_cleaned.csv"
    pq_path = out_dir / "behavior_cleaned.parquet"

    df.to_csv(csv_path, index=False)
    try:
        # Parquet is optional; require pyarrow
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    # Lightweight report
    print("\nPreprocessing complete.")
    print(f"Rows: {len(df):,} | Children: {df['child_id'].nunique()} | Date span: {df['day_utc'].min().date()} → {df['day_utc'].max().date()}")
    print(f"Saved: {csv_path}")
    if pq_path.exists():
        print(f"Saved: {pq_path}")


if __name__ == "__main__":
    main()
