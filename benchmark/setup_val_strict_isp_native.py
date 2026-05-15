from __future__ import annotations

import pathlib
import shlex
import subprocess
import sys
import sysconfig


def _split_flags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("EXT_SUFFIX is unavailable from sysconfig")

    output_path = root / f"_val_strict_isp_native{ext_suffix}"
    source_path = root / "val_strict_isp_native.c"
    include_dir = sysconfig.get_config_var("INCLUDEPY")
    if not include_dir:
        raise RuntimeError("INCLUDEPY is unavailable from sysconfig")

    ldshared = _split_flags(sysconfig.get_config_var("LDSHARED"))
    if not ldshared:
        raise RuntimeError("LDSHARED is unavailable from sysconfig")

    cflags = _split_flags(sysconfig.get_config_var("CFLAGS"))
    command = [
        *ldshared,
        *cflags,
        f"-I{include_dir}",
        str(source_path),
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
