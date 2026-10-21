# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Short-lived CLI client for local-sr.

Loads the OpenVINO RealESRGAN_x4plus pipeline directly in this process,
runs inference, and exits. There is no persistent server.

Exit codes:
  0 - success
  1 - general error (bad args, upscale failure, unsupported hardware)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import colorama

SKILL_NAME = "local-sr"
SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
MODELS_ROOT = OPENVINO_ROOT / "models"
INFO_JSON = Path(__file__).resolve().parent.parent / "info.json"


def _configure_stream_encoding(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


_configure_stream_encoding(sys.stdout)
_configure_stream_encoding(sys.stderr)
colorama.init()


def _normalize_log_path(log_path: str | None) -> Path | None:
    if not log_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = f"~/.openvino/log/local-sr-client-py-{timestamp}.log"
    path = Path(log_path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _log_message(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{timestamp}] [client pid={os.getpid()}] {message}\n")
    except OSError:
        pass


def _load_model_infos() -> list:
    from model_download import load_model_infos
    return load_model_infos(INFO_JSON)


_PROGRESS_STATE: dict = {}


def _format_download_progress(progress) -> str:
    try:
        from model_download import format_download_progress
    except Exception:  # noqa: BLE001 — progress display must never break the client
        return "模型下载中..."
    return format_download_progress(progress)


def _should_display_progress(state) -> bool:
    try:
        from model_download import should_display_progress
    except Exception:  # noqa: BLE001 — throttling must never break the client
        return True
    return should_display_progress(state)


def _report_download_progress(progress, log_path: Path | None = None, force: bool = False) -> None:
    if progress is not None:
        _PROGRESS_STATE["progress"] = progress
    elif not force:
        return
    line = _format_download_progress(progress or _PROGRESS_STATE.get("progress"))
    if not force and line == _PROGRESS_STATE.get("line"):
        return
    # First line shows immediately, then at most one line per 5 minutes, so a
    # multi-GB download reports in periodically instead of flooding the caller.
    if not force and not _should_display_progress(_PROGRESS_STATE):
        return
    _PROGRESS_STATE["line"] = line
    print(line, flush=True)
    _log_message(log_path, line)


def _ensure_model_downloaded(log_path: Path | None) -> None:
    """Synchronously download and validate all required model files."""
    from model_download import ensure_models

    def _download_logger(message: str) -> None:
        print(message, flush=True)
        _log_message(log_path, message)

    ensure_models(
        _load_model_infos(),
        MODELS_ROOT,
        logger=_download_logger,
        progress_callback=lambda progress: _report_download_progress(progress, log_path),
    )


def _load_engine(log_path: Path | None):
    """Build and return a RealESRGANEngine on GPU (with CPU fallback)."""
    from realesrgan_engine import EngineConfig, RealESRGANEngine

    model_infos = _load_model_infos()
    model_info = model_infos[0]
    model_xml = MODELS_ROOT / model_info.dir_name / model_info.required_files[0]

    engine = RealESRGANEngine(
        EngineConfig(
            model_xml=model_xml,
            device="AUTO",  # prefer GPU, fall back to CPU
            tile_size=256,
            tile_pad=10,
        )
    )
    _log_message(log_path, f"loading engine from {model_xml}")
    engine.ensure_compiled()
    _log_message(log_path, "engine ready")
    return engine


def _print_upscale_reply(reply: dict) -> int:
    if not reply.get("ok", False):
        print(colorama.Fore.RED + "❌ 图片放大失败:" + colorama.Style.RESET_ALL)
        print(reply.get("error", reply.get("error_code", "<unknown error>")))
        return 1

    total = reply.get("total", len(reply.get("results", [])))
    succeeded = reply.get("succeeded", total)
    elapsed = reply.get("elapsed_s")
    scale = reply.get("scale")

    if total > 1:
        print(colorama.Style.BRIGHT + colorama.Fore.GREEN + f"\n✅ 批量放大完成: {succeeded}/{total} 成功" + colorama.Style.RESET_ALL)
        if elapsed is not None:
            print(f"总耗时: {float(elapsed):.1f}s")
        output_dir = reply.get("output_dir")
        if output_dir:
            print(f"输出目录: {output_dir}")
    else:
        first = (reply.get("results") or [{}])[0]
        out_path = first.get("output", "")
        in_size = first.get("input_size") or [0, 0]
        out_size = first.get("output_size") or [0, 0]
        per_elapsed = first.get("elapsed_s")
        print(colorama.Style.BRIGHT + colorama.Fore.GREEN + "\n✅ 放大完成" + colorama.Style.RESET_ALL)
        if per_elapsed is not None:
            print(f"放大耗时: {float(per_elapsed):.1f}s")
        if in_size and out_size:
            print(f"输入 {in_size[0]}x{in_size[1]} -> 输出 {out_size[0]}x{out_size[1]} (x{scale})")
        if out_path:
            print(f"输出文件: {out_path}")
    return 0 if reply.get("ok") and succeeded == total else 1


def _upscale_single(file: Path, output_dir: str, scale: int,
                    engine, log_path: Path | None) -> int:
    import cv2

    _log_message(log_path, f"upscale {file} (scale={scale})")
    img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if img is None:
        print(colorama.Fore.RED + f"❌ 无法读取图片: {file}" + colorama.Style.RESET_ALL)
        return 1

    h, w = img.shape[:2]
    t0 = time.time()
    try:
        out = engine.upscale(img, scale)
    except Exception as exc:
        import traceback
        err = f"{exc}\n{traceback.format_exc()}"
        _log_message(log_path, f"inference failed: {err}")
        print(colorama.Fore.RED + "❌ 推理失败:" + colorama.Style.RESET_ALL)
        print(err)
        return 1
    infer_time = time.time() - t0

    out_name = f"{file.stem}_x{scale}{file.suffix}"
    out_path = Path(output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out):
        print(colorama.Fore.RED + f"❌ 无法写入输出文件: {out_path}" + colorama.Style.RESET_ALL)
        return 1

    _log_message(log_path, f"Upscaled {file.name} ({w}x{h} -> {out.shape[1]}x{out.shape[0]}) in {infer_time:.1f}s -> {out_path}")
    reply = {
        "ok": True,
        "results": [{
            "input": str(file),
            "output": str(out_path),
            "input_size": [w, h],
            "output_size": [out.shape[1], out.shape[0]],
            "elapsed_s": infer_time,
            "scale": scale,
            "ok": True,
        }],
        "total": 1,
        "succeeded": 1,
        "output_dir": str(output_dir),
        "elapsed_s": infer_time,
        "scale": scale,
    }
    return _print_upscale_reply(reply)


def _upscale_batch(files: list[Path], output_dir: str, scale: int,
                   engine, log_path: Path | None) -> int:
    import cv2

    print(f"批量放大: 共找到 {len(files)} 个文件，输出目录: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    t0_total = time.time()

    for idx, f in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] 放大: {f.name}")
        _log_message(log_path, f"batch [{idx}/{len(files)}]: {f}")
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            err = f"无法读取图片: {f}"
            print(colorama.Fore.RED + f"  失败: {err}" + colorama.Style.RESET_ALL)
            failed.append((f.name, err))
            continue

        h, w = img.shape[:2]
        t0 = time.time()
        try:
            out = engine.upscale(img, scale)
        except Exception as exc:
            import traceback
            err = f"{exc}"
            _log_message(log_path, f"batch [{idx}/{len(files)}] failed: {f}: {err}\n{traceback.format_exc()}")
            print(colorama.Fore.RED + f"  失败: {err}" + colorama.Style.RESET_ALL)
            failed.append((f.name, err))
            continue
        infer_time = time.time() - t0

        out_name = f"{f.stem}_x{scale}{f.suffix}"
        out_path = Path(output_dir) / out_name
        if not cv2.imwrite(str(out_path), out):
            err = f"无法写入输出文件: {out_path}"
            print(colorama.Fore.RED + f"  失败: {err}" + colorama.Style.RESET_ALL)
            failed.append((f.name, err))
            continue

        print(
            f"  完成: {w}x{h} -> {out.shape[1]}x{out.shape[0]}, "
            f"耗时 {infer_time:.1f}s"
        )
        print(f"  输出文件: {out_path}")
        _log_message(log_path, f"batch [{idx}/{len(files)}] done: {f} -> {out_path}")

    success_count = len(files) - len(failed)
    elapsed_total = time.time() - t0_total
    print(f"\n批量放大完成: {success_count}/{len(files)} 成功")
    if failed:
        print(colorama.Fore.RED + f"以下文件放大失败 ({len(failed)}):" + colorama.Style.RESET_ALL)
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


def _cmd_request(input_path: str, output_dir: str, scale: int,
                 log_path: Path | None = None) -> int:
    resolved = Path(input_path).expanduser()
    if not resolved.exists():
        print(colorama.Fore.RED + f"错误: 路径不存在: {resolved}" + colorama.Style.RESET_ALL)
        return 1
    resolved = resolved.resolve()

    _log_message(
        log_path,
        f"processing request: input={str(resolved)!r} output={output_dir!r} scale={scale}",
    )

    is_dir = resolved.is_dir()
    if is_dir:
        files = sorted(
            f for f in resolved.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(colorama.Fore.RED + f"错误: 目录中未找到可放大的文件（支持: {exts}）" + colorama.Style.RESET_ALL)
            print(f"  目录: {resolved}")
            return 1
        batch_output = output_dir or str(resolved / "output")
    else:
        files = [resolved]
        batch_output = output_dir or str(resolved.parent / "output")
    Path(batch_output).mkdir(parents=True, exist_ok=True)

    try:
        _ensure_model_downloaded(log_path)
        engine = _load_engine(log_path)
    except Exception as exc:
        import traceback
        _log_message(log_path, f"failed to load engine: {exc}\n{traceback.format_exc()}")
        print(colorama.Fore.RED + "❌ 模型加载失败:" + colorama.Style.RESET_ALL)
        print(exc)
        return 1

    try:
        if is_dir:
            return _upscale_batch(files, batch_output, scale, engine, log_path)
        return _upscale_single(files[0], batch_output, scale, engine, log_path)
    finally:
        engine = None
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local-sr CLI client")
    parser.add_argument(
        "-i", "--input",
        type=str, default=None,
        help="input image file or a folder to batch-upscale",
    )
    parser.add_argument(
        "-o", "--output",
        type=str, default="",
        help="output directory (defaults to <input>/output)",
    )
    parser.add_argument(
        "--scale",
        type=int, default=4,
        help="upscale factor (1/2/3/4, default: 4)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="append debug logs to this file",
    )
    args = parser.parse_args(argv)
    log_path = _normalize_log_path(args.log)

    if log_path is not None:
        _log_message(log_path, f"client started with argv={sys.argv[1:]}")

    if args.input is None:
        parser.error("--input is required")
    if args.scale not in (1, 2, 3, 4):
        parser.error("--scale must be one of 1, 2, 3, 4")
    exit_code = _cmd_request(args.input, args.output, args.scale, log_path)

    _log_message(log_path, f"client exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
