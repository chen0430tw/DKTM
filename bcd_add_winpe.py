#!/usr/bin/env python3
"""
bcd_add_winpe.py — DKTM 干净 WinPE BCD 注册工具
================================================

在 bcdedit 被 ACL 封锁的受限环境下，通过直接写 C:\\Boot\\BCD，
将干净 WinPE 注册为固定 GUID 的启动条目。

流程：
  1. 自动发现 ADK winpe.wim（或使用 --wim 指定）
  2. 将 winpe.wim 复制到与 Winre.wim 同一目录
     （设备描述符只需替换文件名，长度不变，二进制结构安全）
  3. 备份 BCD
  4. 启用 SeBackupPrivilege + SeRestorePrivilege
  5. reg load → 克隆 WinRE 设备描述符并替换文件名 → reg unload
  6. 稽核三不变量；失败自动还原

固定 GUID : {7619dcc9-fafe-11d9-b411-000476eba25f}
克隆来源  : {300209a8-6279-11e6-90e0-000c295c2276} (WinRE)

用法:
    python bcd_add_winpe.py                        # 自动发现 ADK winpe.wim
    python bcd_add_winpe.py --wim D:\\custom.wim   # 指定 wim（必须与 Winre.wim 等长文件名）
    python bcd_add_winpe.py --dry-run              # 模拟
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import shutil
import subprocess
import sys

import winreg

# ── 复用 platform_windows ────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from dktm.platform_windows import PlatformOps  # noqa: E402

# ── 常量 ─────────────────────────────────────────────────────────────────────

DKTM_PE_GUID  = "{7619dcc9-fafe-11d9-b411-000476eba25f}"
WINRE_GUID    = "{300209a8-6279-11e6-90e0-000c295c2276}"

TMP_HIVE      = r"HKLM\TmpDKTMPE"
HIVE_SHORT    = "TmpDKTMPE"

OSLOADER_TYPE = 0x10200003

E_DEVICE      = "11000001"   # REG_BINARY  (device descriptor)
E_DESCRIPTION = "12000004"   # REG_SZ      (boot entry name)
E_OSDEVICE    = "21000001"   # REG_BINARY  (device descriptor)
E_SYSTEMROOT  = "22000002"   # REG_SZ      (\Windows)
E_PATH        = "12000002"   # REG_SZ      BcdLibraryString_ApplicationPath (bcdedit "path")
E_WINPE       = "26000022"   # REG_BINARY  BcdOSLoaderBoolean_WinPEMode (1 byte 0x01)
E_DETECTHAL   = "26000010"   # REG_BINARY  BcdOSLoaderBoolean_DetectKernelAndHal (1 byte 0x01)
E_DISPORDER   = "24000001"   # REG_BINARY  (sequence of 16-byte GUIDs)

# ADK 候选安装路径
ADK_ROOTS = [
    r"C:\Program Files (x86)\Windows Kits\10",
    r"C:\Program Files\Windows Kits\10",
]

log = logging.getLogger("bcd_add_winpe")


# ── ADK / wim 发现 ────────────────────────────────────────────────────────────

def _find_adk_winpe_wim() -> str | None:
    """在 ADK 安装目录下寻找 amd64 winpe.wim。"""
    for root in ADK_ROOTS:
        candidate = os.path.join(
            root,
            "Assessment and Deployment Kit",
            "Windows Preinstallation Environment",
            "amd64", "en-us", "winpe.wim",
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def _extract_winre_dir(dev_bytes: bytes) -> str | None:
    """从 WinRE 设备描述符二进制中提取 Winre.wim 所在目录（卷内路径转绝对路径）。

    设备描述符以 UTF-16LE 嵌入卷相对路径，如 \\Recovery\\WindowsRE\\Winre.wim。
    用系统盘盘符拼接即得绝对路径。
    """
    decoded = dev_bytes.decode("utf-16-le", errors="ignore")
    # 找包含 .wim 的路径片段
    for tok in decoded.split("\x00"):
        tok = tok.strip()
        if tok.lower().endswith(".wim") and "\\" in tok:
            # 卷相对路径（\Recovery\WindowsRE\Winre.wim）→ 绝对路径
            sys_drive = os.environ.get("SystemDrive", "C:")
            abs_path = sys_drive + tok
            return os.path.dirname(abs_path)
    return None


def _resolve_wim(user_wim: str | None, dev_bytes: bytes) -> tuple[str, str]:
    """确定 wim 源路径和目标路径。

    目标路径 = Winre.wim 所在目录 / winpe.wim
    （使文件名与 Winre.wim 等长，保证设备描述符二进制原地替换安全）

    Returns
    -------
    (src_wim, dst_wim)
        src_wim : 要使用的 wim 源文件
        dst_wim : 复制到目标位置后的路径（BCD 将指向此处）
    """
    winre_dir = _extract_winre_dir(dev_bytes)
    if winre_dir is None:
        raise RuntimeError("无法从 WinRE 设备描述符提取 Winre.wim 目录")
    log.info("WinRE 目录: %s", winre_dir)

    # 目标文件名与 Winre.wim 等长：9 字符 = 9 字符
    dst_wim = os.path.join(winre_dir, "winpe.wim")

    if user_wim:
        src_wim = user_wim
    else:
        src_wim = _find_adk_winpe_wim()
        if src_wim is None:
            raise RuntimeError(
                "未找到 ADK winpe.wim，请安装 Windows ADK 或用 --wim 指定路径\n"
                f"  搜索路径: {ADK_ROOTS}"
            )
        log.info("自动发现 ADK winpe.wim: %s", src_wim)

    if not os.path.isfile(src_wim):
        raise RuntimeError(f"wim 文件不存在: {src_wim}")

    return src_wim, dst_wim


# ── 注册表写入（REG_OPTION_BACKUP_RESTORE）───────────────────────────────────

def _reg_create(subkey: str) -> int:
    adv = ctypes.windll.advapi32
    adv.RegCreateKeyExW.restype = ctypes.c_long
    hkey = ctypes.c_void_p()
    disp = ctypes.c_ulong()
    ret = adv.RegCreateKeyExW(
        ctypes.c_void_p(0x80000002), subkey, 0, None,
        4,           # REG_OPTION_BACKUP_RESTORE
        0x000F003F,  # KEY_ALL_ACCESS
        None, ctypes.byref(hkey), ctypes.byref(disp),
    )
    if ret != 0:
        raise RuntimeError(f"RegCreateKeyExW({subkey!r}) => {ret}")
    return hkey.value


def _reg_close(hkey: int) -> None:
    ctypes.windll.advapi32.RegCloseKey(hkey)


def _set_binary(hkey: int, name: str, data: bytes) -> None:
    adv = ctypes.windll.advapi32
    adv.RegSetValueExW.restype = ctypes.c_long
    ret = adv.RegSetValueExW(hkey, name, 0, 3, data, len(data))
    if ret != 0:
        raise RuntimeError(f"RegSetValueExW binary({name!r}) => {ret}")


def _set_dword(hkey: int, name: str, value: int) -> None:
    adv = ctypes.windll.advapi32
    adv.RegSetValueExW.restype = ctypes.c_long
    buf = ctypes.c_ulong(value)
    ret = adv.RegSetValueExW(
        hkey, name, 0, 4,
        ctypes.cast(ctypes.byref(buf), ctypes.c_char_p), 4,
    )
    if ret != 0:
        raise RuntimeError(f"RegSetValueExW dword({name!r}) => {ret}")


def _set_sz(hkey: int, name: str, value: str) -> None:
    adv = ctypes.windll.advapi32
    adv.RegSetValueExW.restype = ctypes.c_long
    buf = (value + "\x00").encode("utf-16-le")
    ret = adv.RegSetValueExW(hkey, name, 0, 1, buf, len(buf))
    if ret != 0:
        raise RuntimeError(f"RegSetValueExW sz({name!r}) => {ret}")


def _set_multi_sz(hkey: int, name: str, values: list) -> None:
    """Write REG_MULTI_SZ (type 7) — BCD object list 元素（如 displayorder）。"""
    adv = ctypes.windll.advapi32
    adv.RegSetValueExW.restype = ctypes.c_long
    buf = ("".join(v + "\x00" for v in values) + "\x00").encode("utf-16-le")
    ret = adv.RegSetValueExW(hkey, name, 0, 7, buf, len(buf))
    if ret != 0:
        raise RuntimeError(f"RegSetValueExW multi_sz({name!r}) => {ret}")


# ── hive 挂载 / 卸载 ─────────────────────────────────────────────────────────

def _hive_load(bcd_path: str) -> None:
    r = subprocess.run(["reg", "load", TMP_HIVE, bcd_path],
                       capture_output=True, encoding="mbcs", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"reg load BCD 失败: {r.stderr.strip()}")
    log.info("✓ BCD hive 已挂载")


def _hive_unload() -> None:
    subprocess.run(["reg", "unload", TMP_HIVE],
                   capture_output=True, encoding="mbcs", errors="replace")
    log.info("✓ BCD hive 已卸载（已落盘）")


# ── 读元素 ────────────────────────────────────────────────────────────────────

def _read_elem(guid: str, elem_id: str) -> bytes:
    """读取 REG_BINARY 元素，返回 bytes。仅用于二进制元素（设备描述符等）。"""
    path = f"{HIVE_SHORT}\\Objects\\{guid}\\Elements\\{elem_id}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
        val, regtype = winreg.QueryValueEx(k, "Element")
    if regtype != winreg.REG_BINARY:
        raise RuntimeError(
            f"Elements\\{elem_id} 不是 REG_BINARY (regtype={regtype})，"
            "请用 _read_elem_sz 读取字符串元素"
        )
    return bytes(val)


def _read_elem_sz(guid: str, elem_id: str) -> str:
    """读取 REG_SZ 元素，返回 str。用于 E_PATH / E_SYSTEMROOT 等字符串元素。"""
    path = f"{HIVE_SHORT}\\Objects\\{guid}\\Elements\\{elem_id}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
        val, regtype = winreg.QueryValueEx(k, "Element")
    if regtype not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
        raise RuntimeError(
            f"Elements\\{elem_id} 不是 REG_SZ/REG_EXPAND_SZ (regtype={regtype})，"
            "请用 _read_elem 读取二进制元素"
        )
    return str(val)


# ── 设备描述符文件名补丁 ──────────────────────────────────────────────────────

def _patch_wim_name(device_bytes: bytes, old_name: str, new_name: str) -> bytes:
    """在设备描述符中原地替换 wim 文件名（UTF-16LE）。

    仅替换文件名部分（不含目录），长度必须一致。
    Winre.wim (9) = winpe.wim (9)，天然等长，安全。
    """
    old_enc = old_name.encode("utf-16-le")
    new_enc = new_name.encode("utf-16-le")
    if len(old_enc) != len(new_enc):
        raise RuntimeError(
            f"文件名字符数不同（{len(old_name)} vs {len(new_name)}），"
            "无法原地替换设备描述符。请使用等长文件名（如 winpe.wim = 9 字符）。"
        )
    if old_enc not in device_bytes:
        raise RuntimeError(
            f"设备描述符中未找到 {old_name!r} 的 UTF-16LE 编码。"
        )
    return device_bytes.replace(old_enc, new_enc, 1)


# ── 稽核（三不变量）─────────────────────────────────────────────────────────

def _verify(ops: PlatformOps, dst_wim: str) -> None:
    """重新挂载 BCD，验证三不变量：
    1. bootmgr 对象存在
    2. displayorder 非空
    3. DKTM WinPE 条目存在且 Type 正确，device 元素含目标 wim 文件名
    """
    r = subprocess.run(
        ["reg", "load", TMP_HIVE, ops._BCD_FILE_PATH],
        capture_output=True, encoding="mbcs", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"稽核：reg load 失败: {r.stderr.strip()}")

    issues = []
    try:
        bootmgr = "{" + ops._BOOTMGR_GUID.strip("{}") + "}"
        bootmgr_path = f"{HIVE_SHORT}\\Objects\\{bootmgr}"

        # 1. bootmgr 对象存在
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, bootmgr_path):
                pass
        except OSError:
            issues.append("bootmgr 对象不存在")

        # 2. displayorder 非空
        disp_path = f"{bootmgr_path}\\Elements\\{E_DISPORDER}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, disp_path) as k:
                val, _ = winreg.QueryValueEx(k, "Element")
            if not val:
                issues.append("displayorder 为空")
        except OSError:
            issues.append("displayorder 元素缺失")

        # 3. DKTM WinPE 条目存在 + Type 正确 + device 含目标 wim 文件名
        desc_path = f"{HIVE_SHORT}\\Objects\\{DKTM_PE_GUID}\\Description"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, desc_path) as k:
                t, _ = winreg.QueryValueEx(k, "Type")
            if t != OSLOADER_TYPE:
                issues.append(f"DKTM PE Type 错误: {t:#x}，期望 {OSLOADER_TYPE:#x}")
        except OSError:
            issues.append(f"DKTM WinPE 条目 {DKTM_PE_GUID} 不存在")

        dev_path = f"{HIVE_SHORT}\\Objects\\{DKTM_PE_GUID}\\Elements\\{E_DEVICE}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dev_path) as k:
                dev_val, dev_type = winreg.QueryValueEx(k, "Element")
            if dev_type != winreg.REG_BINARY:
                issues.append(f"device 元素类型错误: {dev_type} (期望 REG_BINARY=3)")
            else:
                wim_enc = os.path.basename(dst_wim).encode("utf-16-le")
                if wim_enc not in bytes(dev_val):
                    issues.append(f"device 元素不含目标 wim 文件名 {os.path.basename(dst_wim)!r}")
        except OSError:
            issues.append("device 元素缺失")

        # 确认 E_PATH / E_SYSTEMROOT 是可读的 REG_SZ
        for label, elem_id in (("path", E_PATH), ("systemroot", E_SYSTEMROOT)):
            ep = f"{HIVE_SHORT}\\Objects\\{DKTM_PE_GUID}\\Elements\\{elem_id}"
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ep) as k:
                    ep_val, ep_type = winreg.QueryValueEx(k, "Element")
                if ep_type not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
                    issues.append(f"{label} 元素类型错误: {ep_type} (期望 REG_SZ=1)")
                elif not ep_val:
                    issues.append(f"{label} 元素为空字符串")
            except OSError:
                issues.append(f"{label} 元素缺失")

    finally:
        subprocess.run(["reg", "unload", TMP_HIVE],
                       capture_output=True, encoding="mbcs", errors="replace")

    if issues:
        raise RuntimeError("BCD 稽核失败: " + "; ".join(issues))

    log.info("✓ 稽核通过（bootmgr / displayorder / DKTM PE 条目）")


# ── 写入 BCD 条目 ─────────────────────────────────────────────────────────────

def _write_entry(ops: PlatformOps, dst_wim: str) -> None:
    """在已挂载 hive 中写入 DKTM WinPE 条目。"""
    dst_wim_name = os.path.basename(dst_wim)   # winpe.wim

    # 从 WinRE 读设备描述符，提取旧 wim 文件名
    dev_raw  = _read_elem(WINRE_GUID, E_DEVICE)
    odev_raw = _read_elem(WINRE_GUID, E_OSDEVICE)

    decoded = dev_raw.decode("utf-16-le", errors="ignore")
    old_wim_name = next(
        (tok.strip() for tok in decoded.split("\x00")
         if tok.strip().lower().endswith(".wim")),
        None,
    )
    if old_wim_name is None:
        old_wim_name = next(
            (seg for seg in decoded.replace("/", "\\").split("\\")
             if seg.lower().endswith(".wim")),
            None,
        )
    if old_wim_name is None:
        raise RuntimeError("无法从 WinRE 设备描述符提取 wim 文件名")

    # 提取结果可能是完整路径（如 \Recovery\WindowsRE\Winre.wim），取 basename
    old_wim_name = os.path.basename(old_wim_name.replace("/", "\\"))
    log.info("WinRE wim 文件名: %s  →  新: %s", old_wim_name, dst_wim_name)
    new_dev  = _patch_wim_name(dev_raw,  old_wim_name, dst_wim_name)
    new_odev = _patch_wim_name(odev_raw, old_wim_name, dst_wim_name)
    log.info("✓ 设备描述符文件名替换完成")

    # 从 WinRE 继承 path / systemroot / detecthal（保证 bootloader 与本机一致）
    # E_PATH 和 E_SYSTEMROOT 是 REG_SZ，必须用 _read_elem_sz / _set_sz
    path_str       = _read_elem_sz(WINRE_GUID, E_PATH)
    systemroot_str = _read_elem_sz(WINRE_GUID, E_SYSTEMROOT)
    try:
        detecthal_raw = _read_elem(WINRE_GUID, E_DETECTHAL)
    except OSError:
        detecthal_raw = b"\x01"

    log.info("继承 path: %s", path_str)
    log.info("继承 systemroot: %s", systemroot_str)

    # 写入新条目
    base = f"{HIVE_SHORT}\\Objects\\{DKTM_PE_GUID}"

    hk = _reg_create(f"{base}\\Description")
    _set_dword(hk, "Type", OSLOADER_TYPE)
    _reg_close(hk)
    log.info("✓ Description\\Type = 0x%08X", OSLOADER_TYPE)

    def _elem(elem_id: str, writer, *args):
        hk = _reg_create(f"{base}\\Elements\\{elem_id}")
        writer(hk, "Element", *args)
        _reg_close(hk)
        log.info("✓ Elements\\%s", elem_id)

    _elem(E_DEVICE,      _set_binary, new_dev)
    _elem(E_OSDEVICE,    _set_binary, new_odev)
    _elem(E_PATH,        _set_sz,     path_str)       # REG_SZ
    _elem(E_SYSTEMROOT,  _set_sz,     systemroot_str)  # REG_SZ
    _elem(E_DETECTHAL,   _set_binary, detecthal_raw)
    _elem(E_WINPE,       _set_binary, b"\x01")
    _elem(E_DESCRIPTION, _set_sz,     "DKTM WinPE")

    # 更新 bootmgr displayorder（REG_MULTI_SZ type=7，存 GUID 字符串列表）
    bootmgr   = "{" + ops._BOOTMGR_GUID.strip("{}") + "}"
    disp_path = f"{HIVE_SHORT}\\Objects\\{bootmgr}\\Elements\\{E_DISPORDER}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, disp_path) as k:
        disp_val, disp_type = winreg.QueryValueEx(k, "Element")

    if disp_type != 7:
        raise RuntimeError(f"displayorder 元素类型非预期: type={disp_type}，期望 REG_MULTI_SZ=7")

    disp_list = list(disp_val)
    if DKTM_PE_GUID.lower() not in [g.lower() for g in disp_list]:
        disp_list.append(DKTM_PE_GUID)
        hk = _reg_create(disp_path)
        _set_multi_sz(hk, "Element", disp_list)
        _reg_close(hk)
        log.info("✓ bootmgr displayorder 已追加 %s", DKTM_PE_GUID)
    else:
        log.info("✓ bootmgr displayorder 已包含该 GUID，跳过")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def add_winpe(user_wim: str | None = None, dry_run: bool = False) -> None:
    ops = PlatformOps(winpe_entry_ids=[], marker_path="", dry_run=dry_run)

    log.info("=== DKTM bcd_add_winpe ===")

    if not ops._bcd_file_available():
        raise RuntimeError(f"BCD 文件不存在: {ops._BCD_FILE_PATH}")

    # 对齐原流程：enable privileges → backup → reg load → write → unload → verify
    # wim 路径在 hive 内读取，与写入合并在同一次 load 会话中完成

    # 1. 启用特权
    ops._enable_backup_restore_privileges()

    # 2. dry-run 模式：仅做路径发现后返回
    if dry_run:
        _hive_load(ops._BCD_FILE_PATH)
        try:
            dev_raw = _read_elem(WINRE_GUID, E_DEVICE)
        finally:
            _hive_unload()
        src_wim, dst_wim = _resolve_wim(user_wim, dev_raw)
        log.info("[DRY-RUN] wim 源: %s", src_wim)
        log.info("[DRY-RUN] wim 目标: %s", dst_wim)
        log.info("[DRY-RUN] 跳过备份 / 写入 / 稽核")
        return

    # 3. 备份（与原流程一致：backup 在 reg load 之前）
    ops._backup_bcd_file()

    # 4. 一次性 reg load，在 hive 内完成读 WinRE + 写新条目
    _hive_load(ops._BCD_FILE_PATH)
    success = False
    dst_wim = None
    try:
        # 读 WinRE 设备描述符，推导 wim 路径
        dev_raw = _read_elem(WINRE_GUID, E_DEVICE)
        src_wim, dst_wim = _resolve_wim(user_wim, dev_raw)
        log.info("wim 源: %s", src_wim)
        log.info("wim 目标: %s", dst_wim)

        # 复制 wim（hive 已挂载，文件操作独立，不冲突）
        if not os.path.exists(dst_wim) or not os.path.samefile(src_wim, dst_wim):
            log.info("复制 winpe.wim → %s ...", dst_wim)
            os.makedirs(os.path.dirname(dst_wim), exist_ok=True)
            shutil.copy2(src_wim, dst_wim)
            log.info("✓ winpe.wim 已就位")
        else:
            log.info("✓ winpe.wim 已存在于目标位置")

        _write_entry(ops, dst_wim)
        success = True
    finally:
        # 与原流程一致：finally 只 unload + log，不在此处还原
        _hive_unload()
        if success:
            log.info("✓ BCD hive 已落盘并卸载")
        else:
            log.warning("BCD hive 已卸载（写入期间发生异常）")

    # 5. 稽核三不变量；失败时还原（与原流程一致）
    try:
        _verify(ops, dst_wim)
    except RuntimeError as exc:
        log.error("稽核失败，从备份还原 BCD: %s", exc)
        ops._restore_bcd_file()
        raise

    log.info("✓ 完成。运行 hot_restart.py --dry-run 确认发现 %s", DKTM_PE_GUID)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DKTM — 将干净 WinPE 注册到 BCD（无需 bcdedit）"
    )
    parser.add_argument(
        "--wim", metavar="PATH", default=None,
        help="PE wim 路径（留空则自动从 ADK 发现）",
    )
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不写入")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if sys.platform != "win32":
        log.error("仅支持 Windows")
        return 1

    try:
        import ctypes as _c
        if not _c.windll.shell32.IsUserAnAdmin():
            log.error("请以管理员身份运行")
            return 1
    except Exception:
        pass

    try:
        add_winpe(user_wim=args.wim, dry_run=args.dry_run)
        return 0
    except Exception as exc:
        log.error("失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
