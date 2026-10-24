from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .config import TRENDS_PATH, ANILIST_PATH, REGIONAL_PATH, MODEL_PATH
from .data_sources import collect_anilist, collect_google_trends
from .modeling import train_model

def main():
    parser = argparse.ArgumentParser(prog="safora", description="SAFORA data and ML CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("collect-anilist")
    a.add_argument("--pages", type=int, default=5)
    a.add_argument("--per-page", type=int, default=50)

    t = sub.add_parser("collect-trends")
    t.add_argument("--titles", required=True, help="Comma-separated title/search terms")
    t.add_argument("--geo", default="IN")
    t.add_argument("--timeframe", default="today 5-y")
    t.add_argument("--sleep", type=float, default=8.0)

    tr = sub.add_parser("train")
    tr.add_argument("--horizon", type=int, default=4)

    args = parser.parse_args()
    if args.command == "collect-anilist":
        df = collect_anilist(ANILIST_PATH, args.pages, args.per_page)
        print(f"Saved {len(df)} metadata rows to {ANILIST_PATH}")
    elif args.command == "collect-trends":
        titles = [x.strip() for x in args.titles.split(",") if x.strip()]
        df = collect_google_trends(titles, TRENDS_PATH, args.geo, args.timeframe, args.sleep)
        print(f"Saved {len(df)} trend observations to {TRENDS_PATH}")
    elif args.command == "train":
        print(train_model(horizon=args.horizon))

if __name__ == "__main__":
    main()
