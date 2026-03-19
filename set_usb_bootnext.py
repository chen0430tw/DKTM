#!/usr/bin/env python3
r"""
set_usb_bootnext.py — 设置 BootNext 指向 Kingston USB (Boot0005)

原理：
  Boot0005 是 Kingston USB 的 UEFI 启动条目，无显式文件路径，
  固件自动搜索 \EFI\Boot\bootx64.efi（EFI fallback 标准路径）。
  我们已将 WinPE 的 bootx64.efi 复制到 E:\EFI\Boot\bootx64.efi，
  因此设置 BootNext=0x0005 即可触发 WinPE 一次性启动。

前提：
  1. E:\EFI\Boot\bootx64.efi  已存在（WinPE bootloader）
  2. E:\sources\boot.wim       已存在（WinPE 镜像）

用法：
  python set_usb_bootnext.py          # 设置 BootNext，手动重启
  python set_usb_bootnext.py --reboot # 设置后自动重启
"""

import ctypes
import struct
import sys
import os
import subprocess

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 常量 ──────────────────────────────────────────────────────────────────────

EFI_GLOBAL_GUID = "{8be4df61-93ca-11d2-aa0d-00e098032b8c}"
BOOT_ENTRY_NUM  = 0x0005   # Boot0005 = Kingston USB
USB_EFI_PATH    = r"E:\EFI\Boot\bootx64.efi"

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY             = 0x0008
SE_PRIVILEGE_ENABLED    = 0x00000002

k32 = ctypes.windll.kernel32
adv = ctypes.windll.advapi32
k32.GetCurrentProcess.restype = ctypes.c_void_p


# ── 权限提升 ──────────────────────────────────────────────────────────────────

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
    proc = k32.GetCurrentProcess()
    if not adv.OpenProcessToken(proc, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                                ctypes.byref(token)):
        raise RuntimeError(f"OpenProcessToken 失败: {k32.GetLastError()}")
    tp = _TP()
    tp.Count = 1
    adv.LookupPrivilegeValueW(None, name, ctypes.byref(tp.Privs[0].Luid))
    tp.Privs[0].Attributes = SE_PRIVILEGE_ENABLED
    adv.AdjustTokenPrivileges(token.value, False, ctypes.byref(tp), 0, None, None)
    k32.CloseHandle(token)


# ── EFI 变量读写 ───────────────────────────────────────────────────────────────

def _read_efi(name: str, guid: str = EFI_GLOBAL_GUID) -> bytes | None:
    k32.GetFirmwareEnvironmentVariableW.restype = ctypes.c_ulong
    buf = ctypes.create_string_buffer(4096)
    r = k32.GetFirmwareEnvironmentVariableW(name, guid, buf, 4096)
    if r == 0:
        return None
    return buf.raw[:r]


def _write_efi(name: str, data: bytes, guid: str = EFI_GLOBAL_GUID,
               attrs: int = 0x00000007) -> None:
    """EFI_VARIABLE_NON_VOLATILE | EFI_VARIABLE_BOOTSERVICE_ACCESS | EFI_VARIABLE_RUNTIME_ACCESS"""
    k32.SetFirmwareEnvironmentVariableExW.restype = ctypes.c_bool
    buf = (ctypes.c_char * len(data))(*data)
    ok = k32.SetFirmwareEnvironmentVariableExW(name, guid, buf, len(data), attrs)
    if not ok:
        raise RuntimeError(f"SetFirmwareEnvironmentVariableExW({name!r}) 失败: {k32.GetLastError()}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def set_usb_bootnext(reboot: bool = False) -> None:
    # 检查 WinPE 文件是否存在于 E:
    if not os.path.isfile(USB_EFI_PATH):
        raise RuntimeError(
            f"WinPE EFI 文件不存在: {USB_EFI_PATH}\n"
            "请先运行: python tools/build_pe.py --output C:\\DKTM_temp\n"
            "然后运行: copy_to_usb.py"
        )
    print(f"[✓] WinPE 文件已确认: {USB_EFI_PATH}")

    print("[*] 启用 SeSystemEnvironmentPrivilege...")
    _enable_privilege("SeSystemEnvironmentPrivilege")
    print("    ✓")

    # 读取当前 BootOrder 以确认 Boot0005 存在
    boot_order = _read_efi("BootOrder")
    if boot_order:
        order = [struct.unpack_from("<H", boot_order, i)[0]
                 for i in range(0, len(boot_order), 2)]
        print(f"[*] 当前 BootOrder: {[f'{x:#06x}' for x in order]}")
        if BOOT_ENTRY_NUM not in order:
            print(f"[!] 警告: Boot{BOOT_ENTRY_NUM:04X} 不在 BootOrder 中！")
            print(f"    将强制设置 BootNext，固件可能忽略。")
    else:
        print("[!] 无法读取 BootOrder")

    # 确认 Boot0005 条目存在
    boot_entry = _read_efi(f"Boot{BOOT_ENTRY_NUM:04X}")
    if boot_entry:
        print(f"[✓] Boot{BOOT_ENTRY_NUM:04X} 存在 ({len(boot_entry)} bytes)")
    else:
        raise RuntimeError(f"Boot{BOOT_ENTRY_NUM:04X} 不存在于 NVRAM 中！")

    print(f"[*] 设置 BootNext = {BOOT_ENTRY_NUM:#06x} (Boot{BOOT_ENTRY_NUM:04X} = Kingston USB)...")
    boot_next_data = struct.pack("<H", BOOT_ENTRY_NUM)
    _write_efi("BootNext", boot_next_data, attrs=0x00000007)
    print(f"    ✓ BootNext 已写入")

    # 验证回读
    bn = _read_efi("BootNext")
    if bn:
        val = struct.unpack("<H", bn)[0]
        ok = "✓" if val == BOOT_ENTRY_NUM else "✗"
        print(f"    验证: BootNext = {val:#06x} {ok}")
    else:
        print("[!] BootNext 回读失败（可能正常，某些固件不回读）")

    print()
    print(f"[✓] EFI NVRAM 已设置")
    print(f"[✓] 下次重启 → 固件加载 Boot{BOOT_ENTRY_NUM:04X} (Kingston USB)")
    print(f"[✓] 固件搜索 E:\\EFI\\Boot\\bootx64.efi → WinPE")
    print(f"[✓] WinPE: wpeinit → wpeutil reboot → 返回 Windows")
    print()

    if reboot:
        print("[*] 5 秒后重启...")
        import time
        for i in range(5, 0, -1):
            print(f"    {i}...", end="\r", flush=True)
            time.sleep(1)
        subprocess.run(["shutdown", "/r", "/t", "0"])


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform != "win32":
        print("仅支持 Windows")
        sys.exit(1)

    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("请以管理员身份运行")
        sys.exit(1)

    reboot = "--reboot" in sys.argv

    try:
        set_usb_bootnext(reboot=reboot)
    except Exception as e:
        print(f"\n[✗] 失败: {e}")
        sys.exit(1)
