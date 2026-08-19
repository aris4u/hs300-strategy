"""Create a portable CPython under runtime\\ so the app does not depend on PATH."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
REQ = ROOT / "requirements.txt"
PY_VER = "3.13.7"
EMBED_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "HS300-setup"})
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"已保存 {dest} ({dest.stat().st_size} bytes)", flush=True)


def runtime_python() -> Path | None:
    p = RUNTIME / "python.exe"
    return p if p.exists() else None


def venv_python() -> Path | None:
    p = ROOT / ".venv" / "Scripts" / "python.exe"
    return p if p.exists() else None


def find_python() -> Path:
    rt = runtime_python()
    if rt:
        return rt
    ve = venv_python()
    if ve:
        return ve
    raise SystemExit("没有 runtime\\python.exe 也没有 .venv。请先运行 tools\\ensure_runtime.py")


def _patch_pth() -> None:
    pth = next(RUNTIME.glob("python*._pth"))
    text = pth.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("#") and "import site" in line:
            lines.append("import site")
            continue
        if line.strip() == "import site":
            lines.append(line)
            continue
        lines.append(line)
    if "import site" not in "\n".join(lines):
        lines.append("import site")
    extra = "Lib\\site-packages"
    if extra not in "\n".join(lines):
        lines.insert(1, extra)
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_from_local_cpython() -> Path | None:
    home = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python313"
    exe = home / "python.exe"
    if not exe.exists():
        return None
    print(f"复制本机 Python 到 runtime：{home}", flush=True)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    shutil.copytree(home, RUNTIME, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "Doc"))
    py = RUNTIME / "python.exe"
    return py if py.exists() else None


def install_embeddable() -> Path:
    copied = install_from_local_cpython()
    if copied:
        return copied
    if runtime_python():
        return runtime_python()  # type: ignore[return-value]
    tmp = ROOT / "tools" / "_cache"
    tmp.mkdir(parents=True, exist_ok=True)
    zpath = tmp / f"python-{PY_VER}-embed-amd64.zip"
    if not zpath.exists() or zpath.stat().st_size < 1_000_000:
        _download(EMBED_URL, zpath)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(RUNTIME)
    _patch_pth()
    pip_py = tmp / "get-pip.py"
    if not pip_py.exists():
        _download(GET_PIP_URL, pip_py)
    py = RUNTIME / "python.exe"
    subprocess.check_call([str(py), str(pip_py), "--no-warn-script-location"])
    return py


def pip_install(py: Path) -> None:
    cmd = [str(py), "-m", "pip", "install", "-r", str(REQ), "--disable-pip-version-check"]
    print("安装依赖 …")
    subprocess.check_call(cmd)


def ensure() -> Path:
    py = runtime_python() or venv_python()
    if py is None:
        print("本机没有可用 Python 环境，开始制作便携 runtime …")
        py = install_embeddable()
        pip_install(py)
        return py
    # Make sure pandas is importable.
    check = subprocess.run([str(py), "-c", "import pandas, matplotlib"], capture_output=True, text=True)
    if check.returncode != 0:
        print("环境不完整，补装依赖 …")
        if py.parent.name.lower() == "runtime":
            pip_install(py)
        else:
            subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(REQ), "--disable-pip-version-check"])
    return py


def main() -> int:
    os.chdir(ROOT)
    force_runtime = "--runtime" in sys.argv
    if force_runtime:
        py = runtime_python()
        if py is None:
            print("制作便携 runtime …")
            py = install_embeddable()
        check = subprocess.run([str(py), "-c", "import pandas, matplotlib"], capture_output=True, text=True)
        if check.returncode != 0:
            pip_install(py)
        print("Python =", py)
        return 0
    py = ensure()
    print("Python =", py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
