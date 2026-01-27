from __future__ import annotations

import argparse
import json
import random

from src.config import load_yaml, parse_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and paths")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = parse_config(raw)

    set_seed(cfg.seed)

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("✅ Config loaded. Dry-run OK.")
        print(json.dumps(raw, indent=2))
        print(f"Output dir: {cfg.paths.output_dir.resolve()}")
        return 0

    # Training will be implemented in the next step
    raise NotImplementedError("Training loop not implemented yet. Run with --dry-run for now.")


if __name__ == "__main__":
    raise SystemExit(main())
