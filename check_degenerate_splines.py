import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Check how many spline rows have degenerate knot vectors (~1)."
    )
    parser.add_argument("path", help="Path to .npy tree file")
    parser.add_argument("--k", type=int, default=39, help="Feature dimension (default: 39)")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.1,
        help="Mean-knot tolerance around 1.0 (default: 0.1)",
    )
    args = parser.parse_args()

    data = np.load(args.path)
    if data.ndim == 1:
        data = data.reshape((-1, args.k))

    if args.k < 39:
        raise SystemExit("k must be >= 39 for spline knots (expected k=39).")

    print(np.max(np.abs(data[:, :3])), np.max(np.abs(data[:, 3:27])))

    knots = data[:, 27:39]
    mean_knots = np.mean(knots, axis=1)
    deg = np.abs(mean_knots - 1.0) < args.tol
    ratio = float(np.mean(deg))

    print(f"degenerate ratio: {ratio:.4f} ({deg.sum()}/{len(deg)})")


if __name__ == "__main__":
    main()
