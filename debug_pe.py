#!/usr/bin/env python3
r"""
debug_pe.py — 构建 Debug 版 WinPE 并部署到 Kingston USB (E:)

Debug WinPE 启动后会：
  1. 将所有执行步骤写入 X:\dktm_debug.log（WinPE 内存盘）
  2. 扫描所有驱动器，找到 E: 上的 DKTM_USB_MARKER 文件（定位 USB）
  3. 将日志复制到 USB:\DKTM\debug.log（重启后可在 Windows 读取）
  4. 屏幕暂停 60 秒，让你看清楚状态
  5. 自动重启回 Windows

用法：
  python debug_pe.py           # 构建 + 部署 + 设置 BootNext，手动重启
  python debug_pe.py --reboot  # 同上 + 自动重启
  python debug_pe.py --read    # 读取上次的 debug 日志 (E:\DKTM\debug.log)
"""

import ctypes
import struct
import sys
import os
import subprocess
import shutil
import logging
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 常量 ──────────────────────────────────────────────────────────────────────

USB_DRIVE        = "E:"
USB_MARKER_FILE  = r"E:\DKTM\DKTM_USB_MARKER.txt"
USB_LOG_FILE     = r"E:\DKTM\debug.log"
USB_SENTINEL     = r"E:\DKTM\boot_sentinel.txt"   # 预写标记，WinPE 追加状态
USB_EFI_PATH     = r"E:\EFI\Boot\bootx64.efi"
TEMP_BUILD_DIR   = r"C:\DKTM_temp_dbg"

EFI_GLOBAL_GUID  = "{8be4df61-93ca-11d2-aa0d-00e098032b8c}"
BOOT_ENTRY_NUM   = 0x0005   # Boot0005 = Kingston USB

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY             = 0x0008
SE_PRIVILEGE_ENABLED    = 0x00000002

k32 = ctypes.windll.kernel32
adv = ctypes.windll.advapi32
k32.GetCurrentProcess.restype = ctypes.c_void_p

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dktm.debug_pe")


# ── Debug startnet.cmd 内容 ───────────────────────────────────────────────────

DEBUG_STARTNET = r"""@echo off
title DKTM Debug WinPE
color 0A

set LOG=X:\dktm_debug.log
echo DKTM Debug WinPE - Log Start > %LOG%
echo Time: %DATE% %TIME% >> %LOG%
echo. >> %LOG%

echo ========================================
echo   DKTM Debug WinPE
echo   Log: %LOG%
echo ========================================
echo.

:: ── Step 1: wpeinit ──────────────────────────────────────────────────────────
echo [STEP 1] Running wpeinit...
echo [STEP 1] wpeinit >> %LOG%
wpeinit
echo [STEP 1] wpeinit exit code: %ERRORLEVEL% >> %LOG%
echo [1] wpeinit done (exit=%ERRORLEVEL%)
echo.

:: ── Step 2: Drive enumeration ─────────────────────────────────────────────────
echo [STEP 2] Enumerating drives...
echo [STEP 2] Drive enumeration >> %LOG%
echo --- Drive letters accessible --- >> %LOG%
for %%D in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if exist %%D:\ (
        echo   Found drive: %%D: >> %LOG%
        echo   [2] Drive %%D: exists
        dir %%D:\ /b /a >>%LOG% 2>nul
    )
)
echo.

:: ── Step 3: Diskpart list volume ──────────────────────────────────────────────
echo [STEP 3] Diskpart volume list...
echo [STEP 3] diskpart list volume >> %LOG%
echo list volume > X:\dp_script.txt
diskpart /s X:\dp_script.txt >> %LOG% 2>&1
echo [3] diskpart done
echo.

:: ── Step 4: Find USB by marker ────────────────────────────────────────────────
echo [STEP 4] Searching for DKTM USB marker...
echo [STEP 4] Looking for DKTM_USB_MARKER.txt >> %LOG%
set USB_FOUND=
for %%D in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if exist %%D:\DKTM\DKTM_USB_MARKER.txt (
        set USB_FOUND=%%D
        echo   USB marker found at %%D: >> %LOG%
        echo [4] USB marker found at %%D:
        goto :usb_found
    )
)
echo [4] USB marker NOT found on any drive >> %LOG%
echo [4] WARNING: USB marker not found - log may not be saved
goto :skip_usb_copy

:usb_found
echo.

:: -- Step 5: Disable network --
echo [STEP 5] Disabling all network adapters...
echo [STEP 5] Disabling network >> %LOG%
echo --- netsh interfaces before --- >> %LOG%
netsh interface show interface >> %LOG% 2>&1
for /f "tokens=1,2,3,4*" %%P in ('netsh interface show interface') do (
    if /i "%%Q"=="Connected" (
        echo   Disabling: [%%S %%T] >> %LOG%
        netsh interface set interface "%%S %%T" disabled >> %LOG% 2>&1
        netsh interface set interface "%%S" disabled >> %LOG% 2>&1
    )
)
echo [5] Network disabled >> %LOG%
echo [5] Network adapters disabled
echo.

:: -- Step 6: Mount Windows partition --
echo [STEP 6] Mounting Windows main partition...
echo [STEP 6] Mounting Windows partition >> %LOG%
echo --- list disk --- >> %LOG%
(echo list disk) > X:\dp2.txt
diskpart /s X:\dp2.txt >> %LOG% 2>&1
echo --- list volume (full) --- >> %LOG%
(echo list volume) > X:\dp3.txt
diskpart /s X:\dp3.txt >> %LOG% 2>&1

:: Try to assign letter W: to volume 1
echo select volume 1 > X:\dp_mount.txt
echo assign letter=W >> X:\dp_mount.txt
echo exit >> X:\dp_mount.txt
diskpart /s X:\dp_mount.txt >> %LOG% 2>&1
ping -n 3 127.0.0.1 > nul

set WIN_DRIVE=
for %%D in (W C D E F G H I J K L M N O P Q R S T U V) do (
    if exist %%D:\Windows\System32\ntoskrnl.exe (
        set WIN_DRIVE=%%D
        echo [6] Windows found at %%D: >> %LOG%
        echo [6] Windows drive: %%D:
        goto :win_found
    )
)
echo [6] Windows partition NOT found >> %LOG%
echo [6] WARNING: Windows partition not found
goto :win_skip

:win_found
echo --- Windows drive root --- >> %LOG%
dir %WIN_DRIVE%:\ /b /a >> %LOG% 2>&1

:: Write test file to check if restore system reverts it
set TESTFILE=%WIN_DRIVE%:\DKTM_restore_test.txt
echo DKTM_TEST %DATE% %TIME% > %TESTFILE% 2>nul
if exist %TESTFILE% (
    echo [6] Test file written: %TESTFILE% >> %LOG%
    echo [6] Test file written to Windows drive OK
) else (
    echo [6] Test file FAILED (write protected?) >> %LOG%
    echo [6] WARNING: Cannot write to Windows drive
)

:win_skip
echo.

:: -- Step 7: System info --
echo [STEP 7] Collecting system info...
echo [STEP 7] System info >> %LOG%
echo --- ipconfig --- >> %LOG%
ipconfig >> %LOG% 2>&1
echo --- environment --- >> %LOG%
set >> %LOG% 2>&1

:: -- Step 8: Append to sentinel --
if not "%USB_FOUND%"=="" (
    echo WINPE_RAN %DATE% %TIME% >> %USB_FOUND%:\DKTM\boot_sentinel.txt
    echo [8] Sentinel updated on %USB_FOUND%:
)

:: -- Step 9: Copy log to USB --
echo [STEP 9] Copying log to USB (%USB_FOUND%:\DKTM\debug.log)...
echo [9] USB=[%USB_FOUND%] LOG=[%LOG%]
if exist %LOG% (echo [9] log file OK) else (echo [9] log file MISSING)
if exist %USB_FOUND%:\DKTM\ (echo [9] DKTM dir OK) else (echo [9] DKTM dir MISSING - mkdir && mkdir %USB_FOUND%:\DKTM\)
echo [STEP 9] Copying log to USB >> %LOG%
copy /y %LOG% %USB_FOUND%:\DKTM\debug.log
if %ERRORLEVEL% == 0 (
    echo [9] Log saved to %USB_FOUND%:\DKTM\debug.log
) else (
    echo [9] ERROR: log copy failed (err=%ERRORLEVEL%)
    echo [9] Trying type redirect...
    type %LOG% > %USB_FOUND%:\DKTM\debug.log
    if %ERRORLEVEL% == 0 (echo [9] type OK) else (echo [9] type also failed)
)

:skip_usb_copy

:: -- Step 10: Pause then reboot --
echo.
echo ========================================
echo   Network: DISABLED
echo   Test file: %TESTFILE%
echo   Log: %USB_FOUND%:\DKTM\debug.log
echo   Rebooting in 60s...
echo ========================================
echo [STEP 10] Waiting 60s >> %LOG%
copy /y %LOG% %USB_FOUND%:\DKTM\debug.log
ping -n 61 127.0.0.1 > nul
wpeutil reboot
"""


# ── EFI 权限 + NVRAM 操作（复用 set_usb_bootnext.py 的实现）────────────────────

def _enable_privilege(name: str) -> None:
    class _LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]
    class _LA(ctypes.Structure):
        _fields_ = [("Luid", _LUID), ("Attributes", ctypes.c_ulong)]
    class _TP(ctypes.Structure):
        _fields_ = [("Count", ctypes.c_ulong), ("Privs", _LA * 1)]
    adv.OpenProcessToken.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)
    ]
    adv.OpenProcessToken.restype = ctypes.c_bool
    token = ctypes.c_void_p()
    adv.OpenProcessToken(k32.GetCurrentProcess(),
                         TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                         ctypes.byref(token))
    tp = _TP(); tp.Count = 1
    adv.LookupPrivilegeValueW(None, name, ctypes.byref(tp.Privs[0].Luid))
    tp.Privs[0].Attributes = SE_PRIVILEGE_ENABLED
    adv.AdjustTokenPrivileges(token.value, False, ctypes.byref(tp), 0, None, None)
    k32.CloseHandle(token)


def _read_efi(name: str) -> bytes | None:
    k32.GetFirmwareEnvironmentVariableW.restype = ctypes.c_ulong
    buf = ctypes.create_string_buffer(4096)
    r = k32.GetFirmwareEnvironmentVariableW(name, EFI_GLOBAL_GUID, buf, 4096)
    return buf.raw[:r] if r else None


def _write_efi(name: str, data: bytes) -> None:
    k32.SetFirmwareEnvironmentVariableExW.restype = ctypes.c_bool
    buf = (ctypes.c_char * len(data))(*data)
    ok = k32.SetFirmwareEnvironmentVariableExW(
        name, EFI_GLOBAL_GUID, buf, len(data), 0x00000007)
    if not ok:
        raise RuntimeError(f"写入 {name} 失败: err={k32.GetLastError()}")


# ── WinPE 构建 ────────────────────────────────────────────────────────────────

def build_debug_pe() -> Path:
    """构建 Debug WinPE，返回 media 目录路径。"""
    sys.path.insert(0, str(Path(__file__).parent / "tools"))
    from build_pe import WinPEBuilder

    out = Path(TEMP_BUILD_DIR)
    builder = WinPEBuilder(str(out))

    log.info("[*] 检测 ADK...")
    if not builder.detect_adk():
        raise RuntimeError("ADK 未找到")

    # 构建前强制清理残留的 DISM 挂载，避免 boot.wim 被锁
    log.info("[*] 清理残留 DISM 挂载...")
    subprocess.run(["dism", "/Cleanup-Wim"], capture_output=True)
    mount_dir = out / "mount"
    if mount_dir.exists():
        subprocess.run(
            ["dism", "/Unmount-Image", f"/MountDir:{mount_dir}", "/Discard"],
            capture_output=True
        )

    log.info("[*] 运行 copype...")
    if not builder.run_copype():
        raise RuntimeError("copype 失败")

    log.info("[*] 挂载 boot.wim...")
    builder.create_mount_point()
    if not builder.mount_wim():
        raise RuntimeError("挂载 boot.wim 失败")

    log.info("[*] 写入 debug startnet.cmd...")
    startnet = out / "mount" / "Windows" / "System32" / "startnet.cmd"
    startnet.write_text(DEBUG_STARTNET, encoding="ascii")
    log.info("    ✓ debug startnet.cmd 已写入")

    log.info("[*] 卸载 boot.wim (commit)...")
    if not builder.unmount_wim(commit=True):
        raise RuntimeError("卸载失败")

    return out / "media"


def deploy_to_usb(media: Path) -> None:
    """将 media/ 整体部署到 E:，保留用户文件。"""
    log.info(f"[*] 部署 WinPE 到 {USB_DRIVE}\\...")

    # 要覆盖的目录（WinPE 相关，不碰用户文件）
    targets = [
        (media / "EFI",     Path(USB_DRIVE) / "EFI"),
        (media / "Boot",    Path(USB_DRIVE) / "Boot"),
        (media / "sources", Path(USB_DRIVE) / "sources"),
    ]
    for src, dst in targets:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        log.info(f"    ✓ {dst}")

    # 创建 USB marker（用于 WinPE 内部定位 USB 驱动器盘符）
    marker_path = Path(USB_MARKER_FILE)
    marker_path.parent.mkdir(exist_ok=True)
    marker_path.write_text("DKTM USB Marker - do not delete\n", encoding="utf-8")
    log.info(f"    ✓ 标记文件: {USB_MARKER_FILE}")

    # 清空上次的 debug 日志和 sentinel，避免读到旧数据
    for f in [USB_LOG_FILE, USB_SENTINEL]:
        p = Path(f)
        if p.exists():
            p.unlink()
    log.info(f"    ✓ 旧日志已清除")


def set_bootnext() -> None:
    """设置 BootNext = 0x0005。"""
    log.info(f"[*] 启用 SeSystemEnvironmentPrivilege...")
    _enable_privilege("SeSystemEnvironmentPrivilege")

    if not _read_efi(f"Boot{BOOT_ENTRY_NUM:04X}"):
        raise RuntimeError(f"Boot{BOOT_ENTRY_NUM:04X} 不存在于 NVRAM")

    log.info(f"[*] 设置 BootNext = {BOOT_ENTRY_NUM:#06x}...")
    _write_efi("BootNext", struct.pack("<H", BOOT_ENTRY_NUM))

    bn = _read_efi("BootNext")
    if bn:
        val = struct.unpack("<H", bn)[0]
        ok = "✓" if val == BOOT_ENTRY_NUM else "✗"
        log.info(f"    验证: BootNext = {val:#06x} {ok}")

    # 写预写标记到 USB（包含时间戳 + BootNext 值）
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentinel = Path(USB_SENTINEL)
    sentinel.parent.mkdir(exist_ok=True)
    sentinel.write_text(
        f"PRE_REBOOT {ts}\n"
        f"BootNext_set = Boot{BOOT_ENTRY_NUM:04X}\n"
        f"Expecting: WINPE_RAN line below if WinPE executes\n",
        encoding="utf-8"
    )
    log.info(f"    ✓ 预写标记: {USB_SENTINEL}")


def read_debug_log() -> None:
    """读取上次启动的 sentinel + debug 日志，诊断启动阶段。"""
    import datetime

    # ── 1. 检查 BootNext 是否已消耗 ──────────────────────────────────────────
    _enable_privilege("SeSystemEnvironmentPrivilege")
    bn = _read_efi("BootNext")
    if bn is None:
        bn_status = f"✓ 已消耗 (err=203) — 固件读取了 BootNext"
    else:
        val = struct.unpack("<H", bn)[0]
        bn_status = f"✗ 未消耗，仍为 {val:#06x} — 固件可能未处理 BootNext"

    # ── 2. 找 USB（扫所有盘找 DKTM_USB_MARKER.txt）────────────────────────────
    usb_drive = USB_DRIVE  # 默认 E:
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        if Path(f"{letter}:\\DKTM\\DKTM_USB_MARKER.txt").exists():
            usb_drive = f"{letter}:"
            break

    sentinel_path = Path(usb_drive) / "DKTM" / "boot_sentinel.txt"
    log_path      = Path(usb_drive) / "DKTM" / "debug.log"
    print(f"[USB] 找到 USB: {usb_drive}  (sentinel={sentinel_path})")

    print(f"\n{'='*60}")
    print("  DKTM Boot Diagnostic")
    print(f"{'='*60}")
    print(f"\n[EFI] BootNext: {bn_status}\n")

    if not sentinel_path.exists():
        print("[SENTINEL] ✗ 预写标记不存在 — 脚本没有成功写入 E: ?")
    else:
        sentinel = sentinel_path.read_text(encoding="utf-8", errors="replace")
        has_winpe = "WINPE_RAN" in sentinel
        print("[SENTINEL] 内容:")
        for line in sentinel.strip().splitlines():
            print(f"    {line}")
        print()
        if has_winpe:
            print("[SENTINEL] ✓ WINPE_RAN 存在 → WinPE startnet.cmd 成功执行")
        else:
            print("[SENTINEL] ✗ 无 WINPE_RAN → WinPE 没有运行（boot 链在 startnet.cmd 之前失败）")
            print()
            print("  可能原因：")
            print("  1. UEFI 无法读取 NTFS (E: 是 NTFS，需要 FAT32)")
            print("  2. EFI\\Microsoft\\Boot\\BCD 路径错误")
            print("  3. boot.wim 损坏或路径不匹配")

    # ── 3. 读 debug log ───────────────────────────────────────────────────────
    print()
    if not log_path.exists():
        print(f"[LOG] ✗ 日志不存在: {log_path}")
        print("      → WinPE 没有运行，或 WinPE 找不到 USB 盘符来写日志")
    else:
        print(f"[LOG] ✓ 日志存在: {log_path}")
        print(f"{'─'*60}")
        print(log_path.read_text(encoding="utf-8", errors="replace"))

    print(f"{'='*60}\n")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if "--read" in sys.argv:
        read_debug_log()
        return

    reboot = "--reboot" in sys.argv

    if sys.platform != "win32":
        print("仅支持 Windows"); sys.exit(1)
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("请以管理员身份运行"); sys.exit(1)

    try:
        log.info("=" * 50)
        log.info("  DKTM Debug PE Builder")
        log.info("=" * 50)

        # 1. 构建
        media = build_debug_pe()
        log.info("    ✓ WinPE 构建完成")

        # 2. 部署到 USB
        deploy_to_usb(media)
        log.info("    ✓ 部署完成")

        # 3. 设置 BootNext
        set_bootnext()

        log.info("")
        log.info("[✓] 全部就绪")
        log.info(f"[✓] BootNext = Boot{BOOT_ENTRY_NUM:04X} (Kingston USB)")
        log.info("[✓] WinPE 将记录日志到 E:\\DKTM\\debug.log")
        log.info("[✓] 重启后进入 WinPE，等待 60 秒后自动返回 Windows")
        log.info("[✓] 返回 Windows 后运行: python debug_pe.py --read")
        log.info("")

        if reboot:
            import time
            log.info("[*] 5 秒后重启...")
            for i in range(5, 0, -1):
                print(f"    {i}...", end="\r", flush=True)
                time.sleep(1)
            subprocess.run(["shutdown", "/r", "/t", "0"])

    except Exception as e:
        print(f"\n[✗] 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
