#!/usr/bin/env python3
"""release_vhd.py — 强制释放 usb.vhd 占用的盘符和挂载"""

import sys, subprocess, ctypes, time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VHD_PATH = r"C:\DKTM_temp_qemu\usb.vhd"


def run_diskpart(commands: str) -> str:
    script = Path(r"C:\DKTM_temp_qemu\dp_release.txt")
    script.parent.mkdir(exist_ok=True)
    script.write_text(commands + "\n", encoding="ascii")
    r = subprocess.run(
        ["diskpart", "/s", str(script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return r.stdout + r.stderr


def main():
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("需要管理员权限"); sys.exit(1)

    if not Path(VHD_PATH).exists():
        print("[✓] usb.vhd 不存在，无需释放")
        return

    print("[*] 强制卸载 usb.vhd...")

    # 第1步：只做 detach，不碰 partition（避免 partition 操作干扰 vdisk 选择）
    out = run_diskpart(
        f'select vdisk file="{VHD_PATH}"\n'
        f'detach vdisk\n'
        f'exit'
    )
    print(out.strip()[-400:] if out.strip() else "(no output)")

    time.sleep(1)

    # 第2步：如果 V: 还在，用 PowerShell 强制移除
    if Path("V:\\").exists():
        print("[!] V: 仍然存在，PowerShell 强制移除...")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Partition | Where-Object {$_.DriveLetter -eq 'V'} | "
             "ForEach-Object { Remove-PartitionAccessPath -DiskNumber $_.DiskNumber "
             "-PartitionNumber $_.PartitionNumber -AccessPath 'V:\\' }"],
            capture_output=True
        )
        time.sleep(1)

    if Path("V:\\").exists():
        print("[!] V: 仍存在")
    else:
        print("[✓] V: 已释放")

    # 第3步：检查文件锁
    try:
        with open(VHD_PATH, 'r+b'):
            pass
        print("[✓] usb.vhd 文件锁已释放，可以删除")
    except PermissionError:
        print("[!] usb.vhd 仍被占用，尝试直接删除...")
        # 最后手段：让 Windows 在重启时删除
        import ctypes as ct
        ok = ct.windll.kernel32.MoveFileExW(VHD_PATH, None, 4)  # MOVEFILE_DELAY_UNTIL_REBOOT
        if ok:
            print("    已标记为重启时删除（MoveFileEx DELAY_UNTIL_REBOOT）")
        else:
            print(f"    标记失败，请手动关闭占用进程后删除: {VHD_PATH}")


if __name__ == "__main__":
    main()
