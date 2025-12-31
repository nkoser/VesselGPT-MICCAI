import argparse
from pipeline import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run modular preprocessing pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--only", default="", help="Comma-separated list of steps to run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    only_steps = None
    if args.only:
        only_steps = [s.strip() for s in args.only.split(",") if s.strip()]

    run_pipeline(cfg, only_steps=only_steps)


if __name__ == "__main__":
    main()
