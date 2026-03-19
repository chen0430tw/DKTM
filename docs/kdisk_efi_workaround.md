# kdisk 写过滤环境下的 EFI NVRAM 解决方案

## 问题背景

### kdisk 写过滤原理

网咖机器通常运行 kdisk 磁盘写过滤驱动。其工作机制如下：

```
[Windows 运行时]
  应用程序 / 文件系统 → kdisk 拦截写操作 → overlay 分区（100MB，磁盘分区2）
                      └─ 读操作：先查 overlay，未命中则读物理磁盘

[重启后，Boot Manager 运行时]
  kdisk 驱动尚未加载 → Boot Manager 直接读物理磁盘扇区
  → 看到的是未修改的原始数据
```

### 为什么原有 BCD 方案失效

`bcd_add_winpe.py` 将 WinPE 条目写入 `C:\Boot\BCD`，`hot_restart.py` 向其中写入 `bootsequence`。这些写操作均被 kdisk 拦截，存入 overlay。重启时：

- Boot Manager 在 kdisk 加载前运行
- 读到的是物理磁盘上**未修改**的 BCD（无 bootsequence，无 DKTM 条目）
- 直接引导回 Windows，WinPE 永远不会启动

### 为什么无法直接写物理磁盘

尝试通过 `\\.\C:` 或 `\\.\PhysicalDrive0` 绕过 kdisk 写入物理扇区时，遭遇两层阻拦：

| 路径 | 失败原因 |
|------|----------|
| `\\.\C:` | Windows 10 1607+ 防止对挂载卷的直接写入（反勒索软件保护），即使启用 `SeManageVolumePrivilege` + `FSCTL_ALLOW_EXTENDED_DASD_IO` 仍拒绝 |
| `\\.\PhysicalDrive0` | 同上，对含已挂载分区的物理盘的扇区写入被系统拒绝 |

---

## 解决方案：EFI NVRAM BootNext

### 关键发现

本机磁盘（Disk 0）为 MBR，但固件为 **UEFI + CSM 模式**。Windows 本身通过 Legacy BIOS 从 MBR 引导，但 UEFI NVRAM 变量仍然完整可用。

```
UEFI NVRAM（固件芯片）
  ↑↓ SetFirmwareEnvironmentVariableExW / GetFirmwareEnvironmentVariableW
  完全不经过磁盘，kdisk 不可见，kdisk 无法拦截
```

### 工作流程

```
set_efi_bootnext.py
  │
  ├─ 启用 SeSystemEnvironmentPrivilege
  ├─ 写入 EFI Boot00FE → D:\DKTM_PE\media\EFI\Boot\bootx64.efi
  └─ 设置 BootNext = 0x00FE（存入固件 NVRAM）

重启
  │
  ▼
UEFI 固件读 BootNext → 加载 Boot00FE
  │
  ▼
D:\DKTM_PE\media\EFI\Boot\bootx64.efi（WinPE 引导器）
  │
  ▼
WinPE startnet.cmd：wpeinit → wpeutil reboot
  │
  ▼
固件自动清除 BootNext → 按 BootOrder 引导 → Windows（MBR Legacy）
```

### 为什么 D: 可用

| 属性 | 说明 |
|------|------|
| 磁盘 | Disk 2，GPT，约 54TB |
| 卷标 | Games1 |
| 持久性 | ✅ 不受 kdisk 写过滤保护，跨重启保留 |
| WinPE 路径 | `D:\DKTM_PE\media\EFI\Boot\bootx64.efi` |

`D:\DKTM_PE` 由 `tools/build_pe.py` 一次性构建，持久存储，每次会话无需重建。

---

## 新增文件说明

### `set_efi_bootnext.py`

每次执行热重启前运行一次，替代原来的 `bcd_add_winpe.py` + BCD bootsequence 方案。

```bash
python set_efi_bootnext.py           # 设置 BootNext，等待手动重启
python set_efi_bootnext.py --reboot  # 设置完直接倒数 5 秒重启
```

**内部流程：**
1. 启用 `SeSystemEnvironmentPrivilege`
2. 从 PowerShell `Get-Partition` 动态获取 D: 的 GPT 分区 GUID、起始 LBA、大小
3. 构造 UEFI EFI_LOAD_OPTION 二进制（HARDDRIVE + FILEPATH 设备路径）
4. 写入 `Boot00FE` EFI 变量
5. 写入 `BootNext = 0x00FE`

**EFI 设备路径格式（短路径，无需 ACPI/PCI 前缀）：**
```
HARDDRIVE(PartNo=2, StartLBA=0x4800, SizeLBA=..., GUID={f10aa8c0-...}, GPT)
→ FILEPATH(\EFI\Boot\bootx64.efi)
→ END
```

### `flush_bcd_physical.py`（已废弃，保留供参考）

尝试通过 `FSCTL_GET_RETRIEVAL_POINTERS` 定位 BCD 物理扇区并直接写入的方案。
由于 Windows 10 的磁盘写保护，最终未能成功，已被 EFI NVRAM 方案取代。

---

## 新的使用流程（受限环境）

### 首次安装（一次性）

```bash
# 1. 构建 WinPE（存到持久盘 D:）
python tools/build_pe.py --output "D:\\DKTM_PE"
```

### 每次使用热重启

```bash
# 设置 EFI BootNext 并重启（两步合一）
python set_efi_bootnext.py --reboot
```

或者用 GUI：（待整合，当前需命令行）

### 与原流程对比

| 项目 | 原流程（BCD） | 新流程（EFI NVRAM） |
|------|--------------|-------------------|
| 每次开机前置步骤 | `bcd_add_winpe.py` | 无（D: 持久保存） |
| 触发热重启 | `hot_restart.py` | `set_efi_bootnext.py --reboot` |
| 跨重启持久性 | ❌ 被 kdisk 吞掉 | ✅ 存在固件 NVRAM |
| 依赖 bcdedit | ❌ 不依赖 | ❌ 不依赖 |
| 一次性启动保证 | BCD bootsequence | EFI BootNext（固件用后自动清除） |

---

## 技术细节

### D: 分区 EFI 设备路径二进制

```
04 01 2A 00                          # HARDDRIVE, Length=42
02 00 00 00                          # PartitionNumber=2
00 48 00 00 00 00 00 00              # StartLBA=0x4800
00 80 1B 49 1B 00 00 00              # SizeLBA
C0 A8 0A F1 2C F7 1A 45              # GPT Partition GUID
A5 BA 32 28 B6 07 C6 90              #   {f10aa8c0-f72c-451a-a5ba-3228b607c690}
02 02                                # GPT format, GPT signature type
04 04 30 00                          # FILEPATH, Length=48
5C 00 45 00 46 00 49 00 5C 00 ...    # \EFI\Boot\bootx64.efi (UTF-16LE)
7F FF 04 00                          # END_OF_PATH
```

### EFI 变量属性

- `Boot00FE`：`NV | BootService | Runtime`（0x00000007），持久存储
- `BootNext`：`NV | BootService | Runtime`（0x00000007），固件引导后自动删除
