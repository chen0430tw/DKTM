#!/usr/bin/env python3
r"""
test_qemu.py — 用 QEMU + OVMF 测试 WinPE USB 启动

将 E: 盘以 raw 模式挂载到 QEMU，模拟 UEFI 从 USB 启动 WinPE。
不需要真机重启，直接在窗口里看 WinPE 是否能启动。

用法：
  python test_qemu.py        # 启动 QEMU 窗口
  python test_qemu.py --dry  # 只打印命令，不执行
"""

import sys
import os
import subprocess
import ctypes
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QEMU_EXE  = r"C:\Program Files\qemu\qemu-system-x86_64.exe"
OVMF_CODE = r"C:\Program Files\qemu\share\edk2-x86_64-code.fd"
OVMF_VARS = r"C:\Program Files\qemu\share\edk2-x86_64-secure-code.fd"

# OVMF vars 需要可写副本（QEMU 会写入 NVRAM 变量）
OVMF_VARS_COPY = r"C:\DKTM_temp_qemu\ovmf_vars.fd"

USB_DRIVE       = "E:"
USB_EFI_CHECK   = r"E:\EFI\Boot\bootx64.efi"
USB_EFI_BCD     = r"E:\EFI\Microsoft\Boot\BCD"


def get_physical_drive_for(drive_letter: str) -> str:
    r"""获取驱动器盘符对应的物理磁盘路径（\\.\PhysicalDriveN）。"""
    import json
    letter = drive_letter.rstrip(":\\")
    out = subprocess.check_output(
        ["powershell", "-command",
         f"Get-Partition -DriveLetter {letter} | "
         f"Get-Disk | Select-Object Number | ConvertTo-Json"],
        encoding="utf-8", errors="replace"
    )
    info = json.loads(out.strip())
    # 可能是单个对象或数组
    if isinstance(info, list):
        num = info[0]["Number"]
    else:
        num = info["Number"]
    return f"\\\\.\\PhysicalDrive{num}"


def make_ovmf_vars_copy() -> str:
    """复制 OVMF vars 到临时目录（QEMU 需要可写副本）。"""
    import shutil
    dst = Path(OVMF_VARS_COPY)
    dst.parent.mkdir(exist_ok=True)
    # 优先用 non-secure vars（无 Secure Boot，兼容性更好）
    # edk2-x86_64-code.fd 对应的 vars 是同目录下没有 secure 的版本
    src_candidates = [
        Path(r"C:\Program Files\qemu\share\edk2-x86_64-vars.fd"),
        Path(OVMF_VARS),
    ]
    src = None
    for c in src_candidates:
        if c.exists():
            src = c
            break
    if src is None:
        raise RuntimeError("找不到 OVMF vars 文件")
    shutil.copy2(src, dst)
    print(f"[*] OVMF vars: {src} → {dst}")
    return str(dst)


def build_qemu_cmd(phys_drive: str, vars_path: str, dry: bool = False) -> list:
    """构造 QEMU 命令行。"""
    # 在 QEMU 里：
    #   -drive if=pflash → UEFI 固件（只读 code + 可写 vars）
    #   -drive file=\\.\PhysicalDriveN,format=raw → USB 物理盘
    #   -boot order=c → 从第一个磁盘启动（我们只挂一个盘）
    cmd = [
        QEMU_EXE,
        "-machine", "q35",               # 现代主板（支持 UEFI）
        "-cpu",     "Westmere",          # 兼容 WinPE 的 CPU
        "-m",       "1G",                # 内存（WinPE 最少需要 512MB）
        "-smp",     "2",                 # 2 个核心
        # UEFI 固件（只读）
        "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
        # UEFI vars（可写副本）
        "-drive", f"if=pflash,format=raw,file={vars_path}",
        # USB 盘（物理磁盘 raw 模式）
        "-drive", f"file={phys_drive},format=raw,if=none,id=usb0",
        "-device", "usb-ehci,id=ehci",
        "-device", "usb-storage,bus=ehci.0,drive=usb0",
        # 显示
        "-vga", "std",
        # 启动顺序：强制从 USB 启动
        "-boot", "order=d,menu=on",
        # 串口日志（方便调试）
        "-serial", "file:C:\\DKTM_temp_qemu\\serial.log",
    ]
    return cmd


def check_prerequisites() -> None:
    """检查前置条件。"""
    errors = []
    if not Path(QEMU_EXE).exists():
        errors.append(f"QEMU 不存在: {QEMU_EXE}")
    if not Path(OVMF_CODE).exists():
        errors.append(f"OVMF code 不存在: {OVMF_CODE}")
    if not Path(USB_EFI_CHECK).exists():
        errors.append(f"USB 上没有 WinPE EFI: {USB_EFI_CHECK}\n"
                      "  请先运行: python debug_pe.py")
    if not Path(USB_EFI_BCD).exists():
        errors.append(f"USB 上没有 UEFI BCD: {USB_EFI_BCD}\n"
                      "  请先运行: python debug_pe.py")
    if errors:
        for e in errors:
            print(f"[✗] {e}")
        sys.exit(1)


def main():
    dry = "--dry" in sys.argv

    if not dry:
        if sys.platform != "win32":
            print("仅支持 Windows"); sys.exit(1)
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("需要管理员权限（读取物理磁盘）"); sys.exit(1)

    print("[*] 检查前置条件...")
    check_prerequisites()
    print("    ✓ QEMU、OVMF、WinPE 文件均存在")

    print(f"[*] 获取 {USB_DRIVE} 的物理磁盘号...")
    if dry:
        phys_drive = r"\\.\PhysicalDrive1"
        print(f"    (dry) 假设: {phys_drive}")
    else:
        phys_drive = get_physical_drive_for(USB_DRIVE)
        print(f"    ✓ {USB_DRIVE} → {phys_drive}")

    print("[*] 准备 OVMF vars 副本...")
    if dry:
        vars_path = OVMF_VARS_COPY
        print(f"    (dry) {vars_path}")
    else:
        vars_path = make_ovmf_vars_copy()

    cmd = build_qemu_cmd(phys_drive, vars_path, dry)

    print()
    print("[*] QEMU 命令:")
    print("    " + " \\\n      ".join(cmd))
    print()

    if dry:
        print("[dry] 不执行，仅打印命令")
        return

    print("[*] 启动 QEMU...")
    print("    窗口打开后等待 WinPE 启动（约 30-60 秒）")
    print("    WinPE 启动后会显示 DKTM Debug 信息，60 秒后自动关机")
    print("    QEMU 串口日志: C:\\DKTM_temp_qemu\\serial.log")
    print()

    # 先 flush E: 写缓冲，避免 QEMU 读到过期数据
    subprocess.run(["powershell", "-command",
                    f"(New-Object System.IO.FileStream('{USB_DRIVE}\\', "
                    f"[System.IO.FileMode]::Open, "
                    f"[System.IO.FileAccess]::Read)).Close()"],
                   capture_output=True)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[*] QEMU 已退出")

    # 读串口日志
    serial_log = Path(r"C:\DKTM_temp_qemu\serial.log")
    if serial_log.exists() and serial_log.stat().st_size > 0:
        print(f"\n[串口日志] {serial_log}:")
        print("─" * 60)
        print(serial_log.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
