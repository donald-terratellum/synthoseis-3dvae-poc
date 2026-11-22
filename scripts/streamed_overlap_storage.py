"""Review fresh-suite storage against pinned inputs; does not launch training."""

import argparse
import json
from pathlib import Path

from scripts.streamed_overlap_epoch_runner import suite_storage_preflight_from_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-record", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    for arm in ("s1", "s2", "s3"):
        parser.add_argument(f"--{arm}-checkpoint-upper-bytes", type=int, required=True)
    parser.add_argument("--other-output-bytes", type=int, required=True)
    parser.add_argument("--free-reserve-bytes", type=int, required=True)
    args = parser.parse_args()
    result = suite_storage_preflight_from_record(
        args.suite_record, args.suite_dir, expected_config_sha256=args.config_sha256,
        checkpoint_upper_bytes={
            "S1": args.s1_checkpoint_upper_bytes, "S2": args.s2_checkpoint_upper_bytes,
            "S3": args.s3_checkpoint_upper_bytes,
        },
        other_output_bytes=args.other_output_bytes, free_reserve_bytes=args.free_reserve_bytes,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
