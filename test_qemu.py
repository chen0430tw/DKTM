#!/usr/bin/env python3
r"""
test_qemu.py — 用 QEMU + OVMF 测试 WinPE 启动（ISO 模式）

将 WinPE 打包成 ISO，挂载到 QEMU 光驱，不需要碰物理 USB 磁盘。
测试 WinPE startnet.cmd 是否正确执行。

用法：
  python test_qemu.py         # 构建 ISO + 启动 QEMU
  python test_qemu.py --dry   # 只打印命令，不执行
"""

import sys
import os
import subprocess
import shutil
import ctypes
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 路径 ──────────────────────────────────────────────────────────────────────

QEMU_EXE   = r"C:\Program Files\qemu\qemu-system-x86_64.exe"
OVMF_CODE  = r"C:\Program Files\qemu\share\edk2-x86_64-code.fd"
OVMF_VARS_SRC = None   # 不存在，直接创建空白 vars（512KB）
OVMF_VARS_SIZE = 512 * 1024   # 512KB = 0x80000

WORK_DIR   = Path(r"C:\DKTM_temp_qemu")
ISO_PATH   = WORK_DIR / "winpe_test.iso"
VARS_COPY  = WORK_DIR / "ovmf_vars.fd"
SERIAL_LOG = WORK_DIR / "serial.log"

# WinPE media 目录（debug_pe.py 构建的结果）
PE_MEDIA   = Path(r"C:\DKTM_temp_dbg\media")



# ── 前置检查 ──────────────────────────────────────────────────────────────────

def check() -> None:
    errors = []
    if not Path(QEMU_EXE).exists():
        errors.append(f"QEMU 不存在: {QEMU_EXE}")
    if not Path(OVMF_CODE).exists():
        errors.append(f"OVMF code 不存在: {OVMF_CODE}")
    if not PE_MEDIA.exists():
        errors.append(
            f"WinPE media 不存在: {PE_MEDIA}\n"
            "  请先运行: python debug_pe.py"
        )
    if errors:
        for e in errors:
            print(f"[✗] {e}")
        sys.exit(1)
    print("    ✓ 所有工具均存在")


# ── 构建 ISO ──────────────────────────────────────────────────────────────────

def build_iso() -> Path:
    """用 MakeWinPEMedia.cmd /ISO 打包 WinPE，保证格式完全正确。"""
    WORK_DIR.mkdir(exist_ok=True)

    make_pe_media = (
        r"C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit"
        r"\Windows Preinstallation Environment\MakeWinPEMedia.cmd"
    )
    if not Path(make_pe_media).exists():
        raise RuntimeError(f"MakeWinPEMedia.cmd 不存在: {make_pe_media}")

    # PE_MEDIA 的父目录就是 copype 的 working directory
    pe_working_dir = PE_MEDIA.parent   # C:\DKTM_temp_dbg

    if ISO_PATH.exists():
        ISO_PATH.unlink()

    # /f = 强制覆盖，不提示确认
    cmd = ["cmd", "/c", make_pe_media, "/ISO", "/f",
           str(pe_working_dir), str(ISO_PATH)]
    print(f"    cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    adk_root = (r"C:\Program Files (x86)\Windows Kits\10"
                r"\Assessment and Deployment Kit")
    env["WinPERoot"]   = adk_root + r"\Windows Preinstallation Environment"
    env["OSCDImgRoot"] = adk_root + r"\Deployment Tools\amd64\Oscdimg"
    env["DISMRoot"]    = adk_root + r"\Deployment Tools\amd64\DISM"
    # oscdimg 需要在 PATH 里才能被 MakeWinPEMedia.cmd 直接调用
    env["PATH"] = adk_root + r"\Deployment Tools\amd64\Oscdimg;" + env.get("PATH", "")

    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-2000:])
    if not ISO_PATH.exists():
        raise RuntimeError(f"MakeWinPEMedia 失败 exit={r.returncode}")
    sz = ISO_PATH.stat().st_size // (1024 * 1024)
    print(f"    ✓ ISO: {ISO_PATH} ({sz} MB)")
    return ISO_PATH


# ── 准备 OVMF vars ────────────────────────────────────────────────────────────

def prepare_vars() -> Path:
    WORK_DIR.mkdir(exist_ok=True)
    # 创建空白 vars（OVMF 首次启动时会自动初始化为默认值）
    VARS_COPY.write_bytes(b"\x00" * OVMF_VARS_SIZE)
    print(f"    ✓ 空白 OVMF vars ({OVMF_VARS_SIZE // 1024}KB) → {VARS_COPY}")
    return VARS_COPY


# ── 构造 QEMU 命令 ────────────────────────────────────────────────────────────

def build_cmd(iso: Path, vars_fd: Path) -> list:
    return [
        QEMU_EXE,
        "-machine", "q35",
        "-cpu",     "qemu64",
        "-m",       "1G",
        "-smp",     "2",
        # UEFI 固件：只读 code + 可写 vars
        "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
        "-drive", f"if=pflash,format=raw,file={vars_fd}",
        # WinPE ISO → IDE CDROM
        "-drive", f"file={iso},media=cdrom,readonly=on,if=ide",
        # 显示
        "-vga", "std",
        # 串口 → 文件（WinPE cmd 输出 + OVMF debug）
        "-chardev", f"file,id=ser0,path={SERIAL_LOG}",
        "-serial",  "chardev:ser0",
    ]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    dry = "--dry" in sys.argv

    print("[*] 检查前置条件...")
    check()

    print("[*] 准备 OVMF vars...")
    vars_fd = prepare_vars() if not dry else VARS_COPY

    print("[*] 构建 WinPE ISO...")
    if dry:
        iso = ISO_PATH
        print(f"    (dry) {iso}")
    else:
        iso = build_iso()

    cmd = build_cmd(iso, vars_fd)

    print()
    print("[*] QEMU 命令:")
    print("    " + "\n      ".join(cmd))
    print()

    if dry:
        print("[dry] 不执行")
        return

    # 清空旧串口日志
    if SERIAL_LOG.exists():
        SERIAL_LOG.unlink()

    print("[*] 启动 QEMU...")
    print("    等待 OVMF logo → WinPE 启动（约 30-60 秒）")
    print("    WinPE 屏幕会显示 DKTM Debug 步骤，60 秒后自动重启")
    print(f"    串口日志: {SERIAL_LOG}")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[*] QEMU 已退出")

    # 读串口日志
    if SERIAL_LOG.exists() and SERIAL_LOG.stat().st_size > 0:
        print(f"\n[串口日志] {SERIAL_LOG}:")
        print("─" * 60)
        content = SERIAL_LOG.read_text(encoding="utf-8", errors="replace")
        print(content[-4000:] if len(content) > 4000 else content)
    else:
        print(f"\n[!] 串口日志为空: {SERIAL_LOG}")
        print("    可能 OVMF 没有向串口输出，属正常现象，看 QEMU 窗口")


if __name__ == "__main__":
    main()
