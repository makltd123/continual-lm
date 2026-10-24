"""Export the learner as a plain Hugging Face model (FR-15): every adapter merged into the weights.

    python export.py --state states/ewai-e5 --out exported/ewai-e5
    python export.py --state states/ewai-e5 --out exported/old --checkpoint ckpt-000012 --dtype float32

The result loads with transformers alone (AutoModelForCausalLM.from_pretrained) — see its README.md.
"""

import argparse

from src.state import export_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", help="default: the newest one")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args(argv)
    print("exported to", export_model(args.state, args.out, args.dtype, args.checkpoint))


if __name__ == "__main__":
    main()
