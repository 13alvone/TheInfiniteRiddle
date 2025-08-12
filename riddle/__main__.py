#!/usr/bin/env python3
from pathlib import Path
import random
from . import parse_args, _resolve_artifact_paths, run_riddle


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    if args.cmd == "generate":
        outdir, db_path = _resolve_artifact_paths(args.outdir, args.db, root)
        theme = None if args.theme == "auto" else args.theme
        run_riddle(
            theme,
            outdir,
            db_path,
            args.bucket,
            args.stems,
            args.mythic_max,
            args.lufs_target,
            args.seed,
            args.verbose,
        )
    elif args.cmd == "spin":
        outdir_arg = args.outdir or "./out"
        db_arg = args.db or "riddle_vault.db"
        theme = args.theme or random.choice(["glass", "salt"])
        bucket = args.bucket or random.choice(["short", "med", "long"])
        stems = args.stems if args.stems is not None else random.choice([True, False])
        mythic_max = args.mythic_max if args.mythic_max is not None else random.randint(0, 3)
        lufs_target = args.lufs_target if args.lufs_target is not None else random.uniform(-18.0, -12.0)
        seed_hex = args.seed or "".join(random.choices("0123456789abcdef", k=32))
        outdir, db_path = _resolve_artifact_paths(outdir_arg, db_arg, root)
        run_riddle(
            theme,
            outdir,
            db_path,
            bucket,
            stems,
            mythic_max,
            lufs_target,
            seed_hex,
            args.verbose,
        )


if __name__ == "__main__":
    main()
