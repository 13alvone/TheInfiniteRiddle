#!/usr/bin/env python3
from pathlib import Path
from . import parse_args, _resolve_artifact_paths, run_riddle


def main() -> None:
    args = parse_args()
    if args.cmd == "generate":
        root = Path.cwd().resolve()
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


if __name__ == "__main__":
    main()
