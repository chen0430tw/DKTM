#!/usr/bin/env python3
"""
set_efi_bootnext.py — 通过 EFI NVRAM BootNext 变量触发 WinPE 一次性启动

原理：
  EFI NVRAM 变量存储在固件芯片中，不在磁盘上，kdisk 写过滤完全看不到。
  本脚本：
    1. 创建/更新 Boot00FE 条目 → D:\DKTM_PE\media\EFI\Boot\bootx64.efi
    2. 设置 BootNext = 0x00FE（一次性，固件用完自动清除）
  下次重启，UEFI 固件直接加载 WinPE；WinPE 执行 wpeinit+wpeutil reboot 返回 Windows；
  之后 BootNext 已清除，正常从 MBR/Windows 启动。

用法：
  python set_efi_bootnext.py          # 设置 BootNext 后等待 hot_restart.py 重启
  python set_efi_bootnext.py --reboot # 设置完直接重启
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
BOOT_ENTRY_NUM  = 0x00FE          # Boot00FE
BOOT_ENTRY_NAME = f"Boot{BOOT_ENTRY_NUM:04X}"

# D: 分区信息（运行时动态获取，此为 fallback 默认值）
WINPE_EFI_PATH  = r"D:\DKTM_PE\media\EFI\Boot\bootx64.efi"
WINPE_FILE_PATH = r"\EFI\Boot\bootx64.efi"   # 分区内路径（反斜杠，UEFI 格式）

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY             = 0x0008
SE_PRIVILEGE_ENABLED    = 0x00000002

k32  = ctypes.windll.kernel32
adv  = ctypes.windll.advapi32
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


# ── 分区信息查询 ───────────────────────────────────────────────────────────────

def _get_d_partition_info():
    """获取 D: 的 GPT 分区 GUID、起始 LBA、大小（扇区数）。"""
    try:
        import subprocess
        out = subprocess.check_output(
            ["powershell", "-command",
             "Get-Partition -DriveLetter D | Select-Object PartitionNumber, Offset, Size, Guid | ConvertTo-Json"],
            encoding="utf-8", errors="replace"
        )
        import json
        info = json.loads(out)
        offset = int(info["Offset"])
        size   = int(info["Size"])
        guid   = info["Guid"].strip("{}")
        partno = int(info["PartitionNumber"])
        return partno, offset // 512, size // 512, guid
    except Exception as e:
        print(f"[!] 自动获取 D: 分区信息失败 ({e})，使用 fallback 值")
        # Fallback: 使用已知值
        return 2, 18432, 117190656000, "f10aa8c0-f72c-451a-a5ba-3228b607c690"


# ── UEFI 设备路径构造 ──────────────────────────────────────────────────────────

def _guid_bytes(guid_str: str) -> bytes:
    """将 GUID 字符串转换为 UEFI 二进制（混合字节序）。"""
    # {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
    g = guid_str.strip("{}")
    parts = g.split("-")
    d1 = int(parts[0], 16)
    d2 = int(parts[1], 16)
    d3 = int(parts[2], 16)
    d4 = bytes.fromhex(parts[3] + parts[4])
    return struct.pack("<IHH", d1, d2, d3) + d4


def _build_dp_harddrive(partno: int, start_lba: int, size_lba: int,
                        part_guid: str) -> bytes:
    """构造 UEFI HARDDRIVE 设备路径节点（GPT 格式）。"""
    # Type=4, SubType=1, Length=42
    guid_bin = _guid_bytes(part_guid)
    return struct.pack(
        "<BBH IQQ 16s BB",
        0x04, 0x01, 42,              # Type, SubType, Length
        partno,                       # PartitionNumber
        start_lba,                    # PartitionStart (LBA)
        size_lba,                     # PartitionSize (sectors)
        guid_bin,                     # PartitionSignature (GPT GUID)
        0x02,                         # PartitionFormat: GPT
        0x02,                         # SignatureType: GPT GUID
    )


def _build_dp_filepath(path: str) -> bytes:
    """构造 UEFI FILE_PATH 设备路径节点。"""
    # path 使用反斜杠（UEFI 格式）
    encoded = (path + "\x00").encode("utf-16-le")
    length = 4 + len(encoded)
    return struct.pack("<BBH", 0x04, 0x04, length) + encoded


def _build_dp_end() -> bytes:
    """构造 END_OF_HARDWARE_DEVICE_PATH 节点。"""
    return struct.pack("<BBH", 0x7F, 0xFF, 4)


def _build_load_option(description: str, dp: bytes) -> bytes:
    """构造完整的 EFI_LOAD_OPTION 二进制。"""
    attrs = struct.pack("<I", 0x00000001)   # LOAD_OPTION_ACTIVE
    desc_bytes = (description + "\x00").encode("utf-16-le")
    fp_len = struct.pack("<H", len(dp))
    return attrs + fp_len + desc_bytes + dp


# ── 主流程 ────────────────────────────────────────────────────────────────────

def set_efi_bootnext(reboot: bool = False) -> None:
    print("[*] 启用 SeSystemEnvironmentPrivilege...")
    _enable_privilege("SeSystemEnvironmentPrivilege")
    print("    ✓")

    # 确认 WinPE EFI 文件存在
    if not os.path.isfile(WINPE_EFI_PATH):
        raise RuntimeError(f"WinPE EFI 文件不存在: {WINPE_EFI_PATH}\n"
                           "请先运行 python tools/build_pe.py --output D:\\DKTM_PE")

    print("[*] 获取 D: 分区信息...")
    partno, start_lba, size_lba, part_guid = _get_d_partition_info()
    print(f"    PartitionNumber={partno}")
    print(f"    StartLBA={start_lba:#x}  SizeLBA={size_lba:#x}")
    print(f"    GUID={{{part_guid}}}")

    print(f"[*] 构造 EFI 设备路径 → {WINPE_FILE_PATH}")
    dp_hdd  = _build_dp_harddrive(partno, start_lba, size_lba, part_guid)
    dp_file = _build_dp_filepath(WINPE_FILE_PATH)
    dp_end  = _build_dp_end()
    dp      = dp_hdd + dp_file + dp_end
    print(f"    DevicePath ({len(dp)} bytes): {dp.hex()}")

    load_opt = _build_load_option("DKTM WinPE", dp)

    print(f"[*] 写入 {BOOT_ENTRY_NAME} → EFI NVRAM...")
    _write_efi(BOOT_ENTRY_NAME, load_opt)
    print(f"    ✓ {BOOT_ENTRY_NAME} 已写入")

    print(f"[*] 设置 BootNext = {BOOT_ENTRY_NUM:#06x}...")
    boot_next_data = struct.pack("<H", BOOT_ENTRY_NUM)
    _write_efi("BootNext", boot_next_data,
               attrs=0x00000007)  # NV|BootService|Runtime
    print(f"    ✓ BootNext = {BOOT_ENTRY_NUM:#06x}")

    # 验证回读
    bn = _read_efi("BootNext")
    if bn:
        val = struct.unpack("<H", bn)[0]
        print(f"    验证: BootNext = {val:#06x} {'✓' if val == BOOT_ENTRY_NUM else '✗'}")

    print()
    print("[✓] EFI NVRAM 已设置，下次重启将引导 WinPE")
    print(f"[✓] WinPE 路径: {WINPE_EFI_PATH}")
    print("[✓] wpeinit + wpeutil reboot → 自动返回 Windows")
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
        set_efi_bootnext(reboot=reboot)
    except Exception as e:
        print(f"\n[✗] 失败: {e}")
        sys.exit(1)
