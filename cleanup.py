#!/usr/bin/env python3
"""
cleanup.py -- 一键清理 DISM 残留挂载 + QEMU 进程 + 临时目录

用法:
  python cleanup.py          # clean all
  python cleanup.py --dism   # DISM only
  python cleanup.py --qemu   # QEMU only
"""
import subprocess
import sys
import shutil
from pathlib import Path

MOUNT_DIR   = Path(r"C:\DKTM_temp_dbg\mount")
BUILD_DIR   = Path(r"C:\DKTM_temp_dbg")
VHD_PATH    = Path(r"C:\DKTM_temp_qemu\usb2.vhd")


def kill_qemu():
    print("[*] Killing QEMU...")
    r = subprocess.run(["taskkill", "/F", "/IM", "qemu-system-x86_64.exe"],
                       capture_output=True, text=True, encoding="mbcs", errors="replace")
    if r.returncode == 0:
        print("    [OK] QEMU killed")
    else:
        print("    [--] No QEMU process")


def detach_vhd():
    print(f"[*] Detaching VHD: {VHD_PATH}")
    if not VHD_PATH.exists():
        print("    [--] VHD not found, skip")
        return
    script = f"select vdisk file={VHD_PATH}\ndetach vdisk\nexit\n"
    r = subprocess.run(["diskpart"], input=script, capture_output=True, text=True, encoding="mbcs", errors="replace")
    if "successfully" in r.stdout.lower() or r.returncode == 0:
        print("    [OK] VHD detached")
    else:
        print(f"    [WARN] diskpart rc={r.returncode}")
        print(r.stdout[-300:])


def cleanup_dism():
    print("[*] Cleaning DISM stale mounts...")
    # 先 cleanup-wim
    r = subprocess.run(["dism", "/Cleanup-Wim"], capture_output=True, text=True, encoding="mbcs", errors="replace")
    print(f"    Cleanup-Wim rc={r.returncode}")

    # 如果 mount 目录还存在，强制 unmount
    if MOUNT_DIR.exists():
        print(f"    Mount dir exists: {MOUNT_DIR}，trying /Unmount-Image /Discard...")
        r2 = subprocess.run(
            ["dism", "/Unmount-Image", f"/MountDir:{MOUNT_DIR}", "/Discard"],
            capture_output=True, text=True, encoding="mbcs", errors="replace"
        )
        print(f"    Unmount rc={r2.returncode}")
        if r2.stdout:
            print("   ", r2.stdout.strip()[-200:])

    # 再跑一次 cleanup-wim 确认干净
    r3 = subprocess.run(["dism", "/Cleanup-Wim"], capture_output=True, text=True, encoding="mbcs", errors="replace")
    print(f"    2nd Cleanup-Wim rc={r3.returncode}")

    print("    [OK] DISM cleanup done")


def delete_build_dir():
    print(f"[*] Deleting build dir: {BUILD_DIR}")
    if not BUILD_DIR.exists():
        print("    [--] Dir not found, skip")
        return
    try:
        shutil.rmtree(BUILD_DIR)
        print("    [OK] Deleted")
    except Exception as e:
        print(f"    [WARN] 删除失败: {e}")
        print("    Trying force delete (rd /s /q)...")
        subprocess.run(["cmd", "/c", f"rd /s /q {BUILD_DIR}"], capture_output=True)
        if not BUILD_DIR.exists():
            print("    [OK] Force delete OK")
        else:
            print("    [ERR] Still exists, may be locked")


def main():
    only_dism = "--dism" in sys.argv
    only_qemu = "--qemu" in sys.argv

    if only_qemu:
        kill_qemu()
        detach_vhd()
        return

    if only_dism:
        cleanup_dism()
        return

    # 默认: all
    kill_qemu()
    detach_vhd()
    cleanup_dism()
    delete_build_dir()
    print("\n[DONE] Done. You can now run debug_pe.py")


if __name__ == "__main__":
    main()
