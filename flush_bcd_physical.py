#!/usr/bin/env python3
"""
flush_bcd_physical.py — 将 C:\Boot\BCD 直接落盘到物理磁盘扇区，绕过 kdisk overlay

原理：
  kdisk 在 Windows 运行时拦截对 C: 的写入，变更只存在于内存/overlay 分区中。
  Boot Manager 在 kdisk 驱动加载之前运行，直接读物理磁盘扇区，看不到 overlay。
  本脚本通过 FSCTL_GET_RETRIEVAL_POINTERS 定位 BCD 文件的物理簇位置，
  再用 \\.\PhysicalDrive0 原始 I/O 写入，使 BCD 修改对 Boot Manager 可见。

用法：
  先运行 bcd_add_winpe.py（写入 overlay），再运行本脚本落盘。
  之后执行 hot_restart.py 时，bootsequence 写入同样需要再次落盘，
  因此 hot_restart.py 完成后本脚本会自动被调用（或手动运行）。
"""

import ctypes
import ctypes.wintypes
import struct
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Win32 常量 ────────────────────────────────────────────────────────────────

GENERIC_READ  = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ  = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

FILE_FLAG_WRITE_THROUGH = 0x80000000
FILE_FLAG_NO_BUFFERING  = 0x20000000

MEM_COMMIT   = 0x00001000
MEM_RESERVE  = 0x00002000
MEM_RELEASE  = 0x00008000
PAGE_READWRITE = 0x04

FSCTL_GET_RETRIEVAL_POINTERS     = 0x00090073
FSCTL_GET_NTFS_VOLUME_DATA       = 0x00090064
IOCTL_DISK_GET_PARTITION_INFO_EX = 0x00070048
FSCTL_ALLOW_EXTENDED_DASD_IO     = 0x00090083   # 允许卷设备扇区级写入

ERROR_MORE_DATA = 38

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY             = 0x0008
SE_PRIVILEGE_ENABLED    = 0x00000002

k32 = ctypes.windll.kernel32
k32.SetFilePointerEx.argtypes = [
    ctypes.c_void_p, ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_ulong,
]
k32.SetFilePointerEx.restype = ctypes.c_bool
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]


# ── 权限提升 ──────────────────────────────────────────────────────────────────

def _enable_privilege(name: str) -> None:
    """启用指定进程特权（同 platform_windows._enable_backup_restore_privileges 模式）。"""
    adv = ctypes.windll.advapi32
    k32.GetCurrentProcess.restype = ctypes.c_void_p

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


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _open(path, access, flags=0):
    h = k32.CreateFileW(
        path, access,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None, OPEN_EXISTING, flags, None,
    )
    if h == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(k32.GetLastError())
    return h


def _ioctl(h, code, in_bytes=b"", out_size=512):
    in_buf = (ctypes.c_char * len(in_bytes))(*in_bytes) if in_bytes else None
    out_buf = ctypes.create_string_buffer(out_size)
    returned = ctypes.c_ulong(0)
    k32.DeviceIoControl(
        h, code,
        in_buf, len(in_bytes),
        out_buf, out_size,
        ctypes.byref(returned), None,
    )
    return out_buf.raw[: returned.value]


def _valloc(size):
    """分配页对齐内存（适用于 NO_BUFFERING 扇区对齐要求）。"""
    ptr = k32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not ptr:
        raise MemoryError(f"VirtualAlloc({size}) 失败")
    return ptr


def _vfree(ptr):
    k32.VirtualFree(ptr, 0, MEM_RELEASE)


# ── 核心查询 ──────────────────────────────────────────────────────────────────

def get_cluster_info(volume=r"\\.\C:"):
    """返回 (bytes_per_sector, bytes_per_cluster)。"""
    h = _open(volume, GENERIC_READ)
    try:
        data = _ioctl(h, FSCTL_GET_NTFS_VOLUME_DATA, out_size=128)
        # NTFS_VOLUME_DATA_BUFFER:
        #   offset 0  : VolumeSerialNumber (8)
        #   offset 8  : NumberSectors (8)
        #   offset 16 : TotalClusters (8)
        #   offset 24 : FreeClusters (8)
        #   offset 32 : TotalReserved (8)
        #   offset 40 : BytesPerSector (4)
        #   offset 44 : BytesPerCluster (4)
        bps = struct.unpack_from("<I", data, 40)[0]
        bpc = struct.unpack_from("<I", data, 44)[0]
        return bps, bpc
    finally:
        k32.CloseHandle(h)


def get_partition_start(volume=r"\\.\C:"):
    """返回分区起始字节偏移（相对于物理磁盘头）。"""
    h = _open(volume, GENERIC_READ)
    try:
        data = _ioctl(h, IOCTL_DISK_GET_PARTITION_INFO_EX, out_size=256)
        # PARTITION_INFORMATION_EX:
        #   offset 0 : PartitionStyle (4) + 4 pad
        #   offset 8 : StartingOffset (LARGE_INTEGER, 8)
        return struct.unpack_from("<q", data, 8)[0]
    finally:
        k32.CloseHandle(h)


def get_retrieval_pointers(filepath):
    """返回文件的物理簇位置列表 [(lcn, vcn_count), ...]。"""
    h = _open(filepath, GENERIC_READ)
    try:
        extents = []
        vcn_input = struct.pack("<q", 0)  # StartingVcn = 0

        while True:
            out_buf = ctypes.create_string_buffer(65536)
            returned = ctypes.c_ulong(0)
            in_buf = (ctypes.c_char * 8)(*vcn_input)
            ok = k32.DeviceIoControl(
                h, FSCTL_GET_RETRIEVAL_POINTERS,
                in_buf, 8,
                out_buf, 65536,
                ctypes.byref(returned), None,
            )
            err = ctypes.get_last_error()
            data = out_buf.raw[: returned.value]

            last_vcn = 0
            if returned.value >= 16:
                # RETRIEVAL_POINTERS_BUFFER:
                #   offset 0  : ExtentCount (4) + 4 pad
                #   offset 8  : StartingVcn (8)
                #   offset 16 : Extents[] { NextVcn(8), Lcn(8) } ...
                n = struct.unpack_from("<I", data, 0)[0]
                prev_vcn = struct.unpack_from("<q", data, 8)[0]
                for i in range(n):
                    base = 16 + i * 16
                    next_vcn = struct.unpack_from("<q", data, base)[0]
                    lcn      = struct.unpack_from("<q", data, base + 8)[0]
                    extents.append((lcn, next_vcn - prev_vcn))
                    prev_vcn = next_vcn
                last_vcn = prev_vcn

            if ok or err != ERROR_MORE_DATA:
                break
            vcn_input = struct.pack("<q", last_vcn)

        return extents
    finally:
        k32.CloseHandle(h)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def flush_bcd_to_physical(
    bcd_path=r"C:\Boot\BCD",
    disk_path=None,           # None = 自动选择
    volume=r"\\.\C:",
):
    print(f"[*] 读取 BCD（overlay 版本）: {bcd_path}")
    with open(bcd_path, "rb") as f:
        bcd_data = f.read()
    file_size = len(bcd_data)
    print(f"    文件大小: {file_size} bytes")

    print("[*] 查询 NTFS 簇信息...")
    bps, bpc = get_cluster_info(volume)
    print(f"    BytesPerSector={bps}  BytesPerCluster={bpc}")

    print("[*] 查询分区起始偏移...")
    part_start = get_partition_start(volume)
    print(f"    分区起始: {part_start:#x} bytes ({part_start // bps} 扇区)")

    print("[*] 查询 BCD 文件物理簇位置...")
    extents = get_retrieval_pointers(bcd_path)
    if not extents:
        raise RuntimeError("未能获取 BCD 文件 extents，无法定位物理位置")
    for lcn, cnt in extents:
        phys = part_start + lcn * bpc
        print(f"    LCN={lcn}  clusters={cnt}  物理偏移={phys:#x}")

    # 策略：
    #   1. 启用 SeManageVolumePrivilege
    #   2. 打开 \\.\C: 并发送 FSCTL_ALLOW_EXTENDED_DASD_IO → 允许扇区级写入
    #      偏移 = lcn * bpc（卷坐标系，不含分区起始）
    #   3. 失败时回退到 \\.\PhysicalDrive0
    #      偏移 = part_start + lcn * bpc（磁盘坐标系）

    print("[*] 启用 SeManageVolumePrivilege...")
    try:
        _enable_privilege("SeManageVolumePrivilege")
        print("    ✓")
    except Exception as e:
        print(f"    警告: {e}（继续尝试）")

    target = disk_path or r"\\.\C:"
    use_part_start = target not in (r"\\.\C:", r"\\.\Volume{whatever}")

    print(f"[*] 打开设备: {target}")
    h = _open(target, GENERIC_READ | GENERIC_WRITE,
              flags=FILE_FLAG_WRITE_THROUGH | FILE_FLAG_NO_BUFFERING)

    # 发送 FSCTL_ALLOW_EXTENDED_DASD_IO，解锁卷设备的扇区级写权限
    ret_bytes = ctypes.c_ulong(0)
    k32.DeviceIoControl(h, FSCTL_ALLOW_EXTENDED_DASD_IO,
                        None, 0, None, 0, ctypes.byref(ret_bytes), None)

    try:
        data_offset = 0
        for lcn, cnt in extents:
            if lcn < 0:  # 稀疏簇，跳过
                data_offset += cnt * bpc
                continue

            byte_offset = (part_start if use_part_start else 0) + lcn * bpc
            chunk_size  = cnt * bpc

            # 取出这段 extents 对应的文件数据，不足整簇则补零
            chunk = bcd_data[data_offset : data_offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))

            # 分配页对齐内存（NO_BUFFERING 要求）
            ptr = _valloc(chunk_size)
            try:
                ctypes.memmove(ptr, chunk, chunk_size)

                ok = k32.SetFilePointerEx(
                    h, ctypes.c_int64(byte_offset), None, 0
                )
                if not ok:
                    raise ctypes.WinError(k32.GetLastError())

                written = ctypes.c_ulong(0)
                ok = k32.WriteFile(
                    h, ctypes.c_void_p(ptr), chunk_size,
                    ctypes.byref(written), None,
                )
                if not ok:
                    raise ctypes.WinError(k32.GetLastError())
            finally:
                _vfree(ptr)

            print(f"    ✓ 写入 {written.value} bytes → 物理偏移 {byte_offset:#x}")
            data_offset += chunk_size

    finally:
        k32.CloseHandle(h)

    print()
    print("[✓] BCD 已落盘到物理磁盘扇区")
    print("[✓] 重启后 Boot Manager 可直接读取（不受 kdisk overlay 影响）")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform != "win32":
        print("仅支持 Windows")
        sys.exit(1)

    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("请以管理员身份运行")
        sys.exit(1)

    try:
        flush_bcd_to_physical()
    except Exception as e:
        print(f"\n[✗] 失败: {e}")
        sys.exit(1)
