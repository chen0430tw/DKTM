#!/usr/bin/env python3
r"""
make_usb_vhd.py — 创建 FAT32 虚拟 USB 磁盘用于 QEMU 测试

创建 GPT + FAT32 的 VHD，复制 WinPE 文件，供 QEMU 当作 USB 设备启动。
OVMF 会走 EFI fallback 路径找 \EFI\Boot\bootx64.efi。

用法：
  python make_usb_vhd.py        # 创建 VHD + 复制文件
  python make_usb_vhd.py --run  # 创建完直接启动 QEMU
"""

import sys, os, subprocess, shutil, time, ctypes
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORK_DIR   = Path(r"C:\DKTM_temp_qemu")
VHD_PATH   = WORK_DIR / "usb2.vhd"
VHD_SIZE   = 600   # MB
PE_MEDIA   = Path(r"C:\DKTM_temp_dbg\media")
QEMU_EXE   = r"C:\Program Files\qemu\qemu-system-x86_64.exe"
OVMF_CODE  = r"C:\Program Files\qemu\share\edk2-x86_64-code.fd"
VARS_COPY  = WORK_DIR / "ovmf_vars.fd"
SERIAL_LOG = WORK_DIR / "serial.log"
OVMF_VARS_SIZE = 512 * 1024


def run(cmd, **kw):
    print(f"    > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, **kw)


def create_vhd():
    """用 diskpart 创建 GPT+FAT32 VHD，挂载，复制文件，卸载。"""
    WORK_DIR.mkdir(exist_ok=True)

    if VHD_PATH.exists():
        VHD_PATH.unlink()
        print(f"    旧 VHD 已删除")

    # diskpart 脚本：只建盘+格式化，不 assign（assign 经常因竞争失败）
    script = f"""create vdisk file="{VHD_PATH}" maximum={VHD_SIZE} type=fixed
select vdisk file="{VHD_PATH}"
attach vdisk
create partition primary
format fs=fat32 quick label=DKTM_PE
exit
"""
    script_path = WORK_DIR / "dp_create.txt"
    script_path.write_text(script, encoding="utf-8")

    print("[*] 创建并挂载 VHD (diskpart)...")
    r = run(["diskpart", "/s", str(script_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1000:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f"diskpart 失败 exit={r.returncode}")

    # 系统 automount 被禁用，无法赋盘符。改用卷 GUID 路径直接访问。
    time.sleep(2)
    print("[*] 获取卷 GUID 路径...")
    r2 = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Volume | Where-Object { $_.FileSystemLabel -eq 'DKTM_PE' } "
         "| Select-Object -ExpandProperty Path"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    vol_path = r2.stdout.strip()
    if not vol_path:
        raise RuntimeError("找不到 DKTM_PE 卷，diskpart 格式化可能失败")
    print(f"    ✓ 卷路径: {vol_path}")
    return vol_path   # 返回给调用方使用


def copy_winpe(vol_path: str = None):
    """复制 WinPE 文件到 VHD（via 卷 GUID 路径，无需盘符）。"""
    if vol_path is None:
        vol_path = "V:\\"
    print(f"[*] 复制 WinPE 文件到 {vol_path}...")
    root = Path(vol_path)
    targets = ["EFI", "Boot", "sources", "bootmgr", "bootmgr.efi"]
    for t in targets:
        src = PE_MEDIA / t
        dst = root / t
        if not src.exists():
            print(f"    跳过（不存在）: {src}")
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"    ✓ {t}")

    # 放 USB 标记文件，让 debug startnet.cmd 能定位 VHD 并保存日志
    dktm_dir = root / "DKTM"
    dktm_dir.mkdir(exist_ok=True)
    (dktm_dir / "DKTM_USB_MARKER.txt").write_text("QEMU_USB_VHD\n", encoding="utf-8")
    print("    ✓ DKTM_USB_MARKER.txt")

    # 验证关键文件
    for f in ["EFI/Boot/bootx64.efi", "EFI/Microsoft/Boot/BCD",
              "sources/boot.wim", "Boot/boot.sdi"]:
        p = root / f
        print(f"    {'✓' if p.exists() else '✗'} {p}")


def detach_vhd():
    """卸载 VHD。"""
    script = f"""select vdisk file="{VHD_PATH}"
detach vdisk
exit
"""
    script_path = WORK_DIR / "dp_detach.txt"
    script_path.write_text(script, encoding="utf-8")
    print("[*] 卸载 VHD...")
    r = run(["diskpart", "/s", str(script_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-500:] if r.stdout else "")
    print("    ✓ VHD 已卸载")


def prepare_vars():
    WORK_DIR.mkdir(exist_ok=True)
    VARS_COPY.write_bytes(b"\x00" * OVMF_VARS_SIZE)
    print(f"    ✓ OVMF vars ({OVMF_VARS_SIZE//1024}KB)")
    return VARS_COPY


def run_qemu():
    """启动 QEMU，以 USB 设备挂载 VHD。"""
    global SERIAL_LOG
    try:
        if SERIAL_LOG.exists():
            SERIAL_LOG.unlink()
    except PermissionError:
        SERIAL_LOG = WORK_DIR / f"serial_{int(time.time())}.log"

    cmd = [
        QEMU_EXE,
        "-machine", "q35",
        "-cpu",     "Nehalem",
        "-m",       "2G",          # 2GB，boot.wim 326MB ramdisk 需要足够内存
        "-smp",     "2",
        # UEFI 固件
        "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
        "-drive", f"if=pflash,format=raw,file={VARS_COPY}",
        # VHD 作为 USB 设备（OVMF 会找 \EFI\Boot\bootx64.efi）
        "-drive", f"file={VHD_PATH},format=vpc,if=none,id=usbdisk",
        "-device", "usb-ehci,id=ehci",
        "-device", "usb-storage,bus=ehci.0,drive=usbdisk",
        "-vga", "std",
        "-chardev", f"file,id=ser0,path={SERIAL_LOG}",
        "-serial",  "chardev:ser0",
        "-no-reboot",
        "-d", "cpu_reset,guest_errors",
        "-D", str(WORK_DIR / "qemu_guest.log"),
    ]

    print("\n[*] 启动 QEMU (USB VHD 模式, 2GB RAM)...")
    print("    等待 OVMF 找到 USB → WinPE 启动（约 1-3 分钟）")
    print(f"    串口: {SERIAL_LOG}\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[*] QEMU 已退出")

    if SERIAL_LOG.exists() and SERIAL_LOG.stat().st_size > 0:
        content = SERIAL_LOG.read_text(encoding="utf-8", errors="replace")
        print(f"\n[串口日志]\n{'─'*60}")
        print(content[-3000:] if len(content) > 3000 else content)

    # 挂载 VHD 读取 WinPE debug 日志
    print(f"\n[*] 读取 WinPE debug 日志...")
    r_attach = subprocess.run(
        ["diskpart", "/s", str(WORK_DIR / "dp_detach.txt")],  # reuse script path
        capture_output=True
    )
    # 写 attach 脚本
    attach_script = WORK_DIR / "dp_read.txt"
    attach_script.write_text(
        f'select vdisk file="{VHD_PATH}"\nattach vdisk readonly\nexit\n',
        encoding="ascii"
    )
    subprocess.run(["diskpart", "/s", str(attach_script)], capture_output=True)
    time.sleep(2)

    # 通过卷 GUID 找日志
    r2 = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Volume | Where-Object { $_.FileSystemLabel -eq 'DKTM_PE' } "
         "| Select-Object -ExpandProperty Path"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    vol = r2.stdout.strip()
    if vol:
        log_path = Path(vol) / "DKTM" / "debug.log"
        if log_path.exists():
            print(f"\n[WinPE debug.log]\n{'─'*60}")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
        else:
            print(f"    [!] debug.log 不存在 ({log_path})")
            print(f"    VHD 上的文件: {list((Path(vol) / 'DKTM').iterdir()) if (Path(vol)/'DKTM').exists() else '无 DKTM 目录'}")
    else:
        print("    [!] 找不到 DKTM_PE 卷")

    # 卸载
    subprocess.run(["diskpart", "/s", str(WORK_DIR / "dp_detach.txt")],
                   capture_output=True)


def main():
    run_only = "--run" in sys.argv

    if sys.platform != "win32":
        print("仅 Windows"); sys.exit(1)
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("需要管理员权限"); sys.exit(1)

    if not PE_MEDIA.exists():
        print(f"[✗] WinPE media 不存在: {PE_MEDIA}")
        print("    请先运行: python debug_pe.py")
        sys.exit(1)

    if not run_only:
        vol_path = create_vhd()
        copy_winpe(vol_path)
        detach_vhd()
        print("\n[✓] VHD 已创建:", VHD_PATH)

    print("[*] 准备 OVMF vars...")
    prepare_vars()
    run_qemu()


if __name__ == "__main__":
    main()
