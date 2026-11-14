#!/usr/bin/env python3
"""Run the Stage 3 benchmark on a connected Android phone, then score it.

    python scripts/run_device_eval.py --serial <adb serial>

Steps:
  1. Push data/raw + manifest + queries into the app's *private* external folder
     (/sdcard/Android/data/com.pixarch.app/files/eval). Media scanning skips that
     folder, so nothing appears in the phone's gallery.
  2. Build and install the debug app and its instrumentation APK (gradlew).
  3. Run DeviceEvalTest with `am instrument` -- not connectedAndroidTest, which
     uninstalls the app (and its results) when it finishes.
  4. Pull eval_out/ (device_run.json + device_index.db) to data/device_runs/<run>/.
  5. Delete the corpus and results from the phone (unless --keep-on-device).
  6. Score with scripts/evaluate_device.py (PC side re-scored from index.db).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch.eval import runner  # noqa: E402

PKG = "com.pixarch.app"
DEVICE_ROOT = f"/sdcard/Android/data/{PKG}/files"
TEST_CLASS = f"{PKG}.eval.DeviceEvalTest"
LOG = "/data/local/tmp/pixarch_device_eval.log"


def find_adb() -> str:
    for base in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")):
        if base:
            exe = Path(base) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if exe.exists():
                return str(exe)
    found = shutil.which("adb")
    if not found:
        raise SystemExit("adb not found; set ANDROID_HOME")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", required=True, help="adb serial (see `adb devices`)")
    ap.add_argument("--skip-push", action="store_true", help="corpus is already on the phone")
    ap.add_argument("--skip-install", action="store_true", help="APKs are already installed")
    ap.add_argument("--keep-on-device", action="store_true", help="leave corpus/results on the phone")
    ap.add_argument("--no-pc", action="store_true", help="skip PC re-scoring")
    ap.add_argument("--collect-only", action="store_true",
                    help="a run already started on the phone: just wait for it, pull and score")
    args = ap.parse_args()
    if args.collect_only:
        args.skip_push = args.skip_install = True

    adb_exe = find_adb()
    env = dict(os.environ, ANDROID_SERIAL=args.serial, MSYS_NO_PATHCONV="1")

    def adb(*a: str, check: bool = True, capture: bool = False, timeout: float | None = None) -> str:
        try:
            r = subprocess.run([adb_exe, "-s", args.serial, *a], env=env, text=True, timeout=timeout,
                               capture_output=capture, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            if check:
                raise SystemExit(f"adb {' '.join(a)} timed out (phone disconnected?)")
            return ""
        if check and r.returncode != 0:
            raise SystemExit(f"adb {' '.join(a)} failed:\n{r.stdout or ''}{r.stderr or ''}")
        return r.stdout or ""

    state = adb("get-state", check=False, capture=True, timeout=15).strip()
    if state != "device":
        raise SystemExit(f"Phone {args.serial} not connected (state: {state or 'none'})")
    model = adb("shell", "getprop", "ro.product.model", capture=True).strip()
    print(f"== phone: {model} ({args.serial})")

    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    qrels = runner.load_qrels(ROOT / cfg["paths"]["qrels"])
    depth = runner.retrieval_depth(qrels, cfg["search"]["top_k"], cfg["eval"]["k_values"])

    if not args.skip_install:
        print("== building + installing app and test APKs")
        gradlew = ROOT / "android" / ("gradlew.bat" if os.name == "nt" else "gradlew")
        r = subprocess.run([str(gradlew), "--no-daemon", "installDebug", "installDebugAndroidTest"],
                           cwd=ROOT / "android", env=env)
        if r.returncode != 0:
            raise SystemExit("gradle install failed")

    if not args.skip_push:
        print("== pushing corpus (data/raw, manifest, queries) to the app's private folder")
        adb("shell", "rm", "-rf", f"{DEVICE_ROOT}/eval")
        adb("shell", "mkdir", "-p", f"{DEVICE_ROOT}/eval")
        t0 = time.time()
        adb("push", str(ROOT / cfg["paths"]["raw_dir"]), f"{DEVICE_ROOT}/eval/raw")
        adb("push", str(ROOT / cfg["paths"]["manifest"]), f"{DEVICE_ROOT}/eval/manifest.csv")
        adb("push", str(ROOT / cfg["paths"]["queries"]), f"{DEVICE_ROOT}/eval/queries.jsonl")
        print(f"   pushed in {time.time() - t0:.0f} s")

    if not args.collect_only:
        runners = [line.split(":", 1)[1].split(" ")[0] for line in
                   adb("shell", "pm", "list", "instrumentation", capture=True).splitlines()
                   if f"(target={PKG})" in line]
        if not runners:
            raise SystemExit("instrumentation APK not installed")
        print(f"== starting {TEST_CLASS} via {runners[0]} (depth {depth}), detached on the phone")
        # Detached with nohup: wireless adb drops when the screen locks or Wi-Fi
        # dozes, and a foreground `am instrument` dies with the adb session.
        adb("shell", "rm", "-rf", f"{DEVICE_ROOT}/eval_out", LOG)
        adb("shell", f"nohup am instrument -w -r -e class {TEST_CLASS} -e depth {depth} "
                     f"{runners[0]} > {LOG} 2>&1 &")

    print("== waiting for the phone to finish (survives adb disconnects; reconnect if it drops)")
    t0 = time.time()
    while True:
        time.sleep(20)
        log = adb("shell", "cat", LOG, check=False, capture=True, timeout=30)
        if "OK (1 test)" in log:
            break
        if "FAILURES!!!" in log or "INSTRUMENTATION_FAILED" in log or "Process crashed" in log:
            print(log[-3000:])
            raise SystemExit("on-device run failed; see output above (and `adb logcat -s DeviceEval`)")
        if not log:
            print(f"   [{time.time() - t0:5.0f} s] phone unreachable -- reconnect wireless debugging; still waiting")
        elif time.time() - t0 > 3 * 3600:
            raise SystemExit("gave up after 3 h")
    print(f"   finished after {time.time() - t0:.0f} s of waiting")

    run_dir = ROOT / "data" / "device_runs" / f"{model.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("device_run.json", "device_index.db"):
        adb("pull", f"{DEVICE_ROOT}/eval_out/{name}", str(run_dir / name))
    print(f"== results pulled to {run_dir.relative_to(ROOT)}")

    if not args.keep_on_device:
        adb("shell", "rm", "-rf", f"{DEVICE_ROOT}/eval", f"{DEVICE_ROOT}/eval_out")
        print("== removed corpus and results from the phone")

    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_device.py"), "--run", str(run_dir)]
    if args.no_pc:
        cmd.append("--no-pc")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
