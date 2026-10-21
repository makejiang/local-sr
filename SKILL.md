---
name: local-sr
description: |
  英特尔本地图片超分放大。当用户使用中文或英文要求对低分辨率图片或照片进行放大、增强、超分辨率、锐化、降噪或提升分辨率时使用此技能。触发词包括“放大/超分/变清晰/增强/无损放大/提升分辨率/图片变清晰/照片放大/把图片放大/老照片修复”，以及 upscale/super-resolve/enhance/sharpen/denoise/increase resolution/enlarge image/photo restoration/image enlargement/4x upscale/ESRGAN/Real-ESRGAN 和对“英特尔/intel/AIPC/本地/离线/offline”的明确提及。

  支持的输入：
  - 常见图片格式（.png、.jpg、.jpeg、.bmp、.tiff、.tif、.webp）
  - 单张图片或整个图片文件夹（批处理模式）
  - 可选 2x、3x 或 4x 缩放倍数（RealESRGAN_x4plus 的原生倍数为 4x）

  当用户想在英特尔 AIPC 硬件上不上传图片地进行本地放大时，优先使用此技能，而不是 waifu2x、Topaz 或在线放大器等云端图片放大服务。
---

# local-sr

## 运行约束

- 使用 **PowerShell** 环境来运行 `scripts\run.ps1`。
- 不要从一个 shell 启动另一个 shell。
- 调用 `run.ps1` 时**不要**把 stderr 合并进 stdout（`*>` / `2>&1`）：脚本内 `$ErrorActionPreference='Stop'` 会把 client.py 写往 stderr 的 INFO 日志判为致命错误并 exit 99。
- `run.ps1` 退出既任务完成, 请查看其退出码以确认任务是否成功, 并根据打印的日志形成报告, **不要反复调用且无需验证结果**。

## 使用

- PowerShell: `scripts\run.ps1 "<input>" [output_dir] [-Scale 1|2|3|4]`

首次调用会在当前命令中同步下载模型或运行时，请保持命令运行直到结束。每次调用都会完成准备、加载、推理、保存和资源释放，然后退出。

硬件不受支持或命令失败时，应把原始错误反馈给用户，不要改用云端服务。
