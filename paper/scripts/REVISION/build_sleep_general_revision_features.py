#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sleep_general_revision_features import (
    load_sleep_general_revision_table,
    write_sleep_general_revision_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform raw brainstorm_spin sleep_general output into the revision master-table schema."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, default=None)
    parser.add_argument("--audit-title", default="Sleep General Revision Features Audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_sleep_general_revision_table(args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    if args.audit_md is not None:
        args.audit_md.parent.mkdir(parents=True, exist_ok=True)
        write_sleep_general_revision_audit(
            df,
            args.audit_md,
            source=args.input_csv,
            title=args.audit_title,
        )
    print(f"Wrote {len(df):,} rows to {args.output_csv}")
    if args.audit_md is not None:
        print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
