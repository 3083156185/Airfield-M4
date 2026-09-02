import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"


@dataclass
class JobResult:
    config: str
    command: List[str]
    returncode: int
    elapsed_seconds: float


def _add_optional_value(command: List[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _add_optional_bool(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_train_command(args: argparse.Namespace, config: str) -> List[str]:
    command = [sys.executable, str(TRAIN_SCRIPT), "-c", config]

    _add_optional_value(command, "--resume", args.resume)
    _add_optional_value(command, "--tuning", args.tuning)
    _add_optional_value(command, "--device", args.device)
    _add_optional_value(command, "--seed", args.seed)
    _add_optional_bool(command, "--use-amp", args.use_amp)
    _add_optional_bool(command, "--use-amp-fallback-fp32", args.use_amp_fallback_fp32)
    _add_optional_value(command, "--output-dir", args.output_dir)
    _add_optional_value(command, "--summary-dir", args.summary_dir)
    _add_optional_bool(command, "--test-only", args.test_only)
    _add_optional_value(command, "--path", args.path)
    _add_optional_value(command, "--mode", args.mode)
    _add_optional_value(command, "--print-method", args.print_method)
    _add_optional_value(command, "--print-rank", args.print_rank)
    _add_optional_value(command, "--local-rank", args.local_rank)

    if args.update:
        command.append("--update")
        command.extend(args.update)

    return command


def build_train_commands(args: argparse.Namespace) -> List[List[str]]:
    return [build_train_command(args, config) for config in args.configs]


def run_jobs(args: argparse.Namespace) -> List[JobResult]:
    results = []
    commands = build_train_commands(args)
    total = len(commands)

    for index, (config, command) in enumerate(zip(args.configs, commands), start=1):
        print(f"[train_auto] start {index}/{total}: {config}", flush=True)
        print(f"[train_auto] command: {shlex.join(command)}", flush=True)

        started_at = time.monotonic()
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        elapsed_seconds = time.monotonic() - started_at

        result = JobResult(
            config=config,
            command=command,
            returncode=completed.returncode,
            elapsed_seconds=elapsed_seconds,
        )
        results.append(result)

        print(
            f"[train_auto] finish {index}/{total}: {config} "
            f"returncode={result.returncode} elapsed={result.elapsed_seconds:.1f}s",
            flush=True,
        )

        if result.returncode != 0 and args.stop_on_fail:
            print("[train_auto] stop-on-fail enabled; remaining configs skipped.", flush=True)
            break

    return results


def _print_dry_run(commands: Iterable[Sequence[str]]) -> None:
    print("[train_auto] DRY-RUN: commands that would be executed:")
    for index, command in enumerate(commands, start=1):
        print(f"[train_auto] {index}: {shlex.join(command)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple DEIM training YAML configs sequentially via train.py."
    )

    parser.add_argument("-c", "--configs", nargs="+", required=True, help="YAML config files to train in order")
    parser.add_argument("--dry-run", action="store_true", help="print train.py commands without running them")
    parser.add_argument("--stop-on-fail", action="store_true", help="stop after the first non-zero train.py exit code")

    parser.add_argument("-r", "--resume", type=str, help="resume checkpoint passed to train.py")
    parser.add_argument("-t", "--tuning", type=str, help="tuning checkpoint passed to train.py")
    parser.add_argument("-d", "--device", type=str, help="device passed to train.py")
    parser.add_argument("--seed", type=int, help="seed passed to train.py")
    parser.add_argument("--use-amp", action="store_true", help="enable train.py AMP")
    parser.add_argument(
        "--use-amp-fallback-fp32",
        action="store_true",
        help="retry AMP NaN forward in FP32 and use FP32 for the rest of the epoch",
    )
    parser.add_argument("--output-dir", type=str, help="output directory passed to train.py")
    parser.add_argument("--summary-dir", type=str, help="summary directory passed to train.py")
    parser.add_argument("--test-only", action="store_true", default=False, help="pass train.py --test-only")
    parser.add_argument("-p", "--path", type=str, help="onnx/engine/prune-model path in train.py test-only")
    parser.add_argument("-m", "--mode", type=str, default=None, choices=["det", "mask"], help="train.py engine mode")
    parser.add_argument("-u", "--update", nargs="+", help="train.py yaml overrides, e.g. a.b=1 c=True")
    parser.add_argument("--print-method", type=str, default=None, help="print method passed to train.py")
    parser.add_argument("--print-rank", type=int, default=None, help="print rank passed to train.py")
    parser.add_argument("--local-rank", type=int, help="local rank passed to train.py")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    commands = build_train_commands(args)

    if args.dry_run:
        _print_dry_run(commands)
        return 0

    try:
        results = run_jobs(args)
    except KeyboardInterrupt:
        print("[train_auto] interrupted by user.", flush=True)
        return 130

    failed = [result for result in results if result.returncode != 0]
    print(
        f"[train_auto] summary: total={len(results)} failed={len(failed)} "
        f"skipped={len(args.configs) - len(results)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
