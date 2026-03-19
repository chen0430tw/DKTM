# DKTM — Dynamic Kernel Transition Mechanism

一键热重启：进入 WinPE 重置 Windows 内核，WinPE 自动返回，全程无需手动操作。

专为 **bcdedit 被 ACL 封锁的受限环境**（如网咖）设计，通过直接写 BCD 文件绕过限制。

---

## 工作原理

```
按下按钮
   │
   ├─ health check（权限 / WinPE / 磁盘）
   ├─ 停止非必要服务（spooler / SysMain / WSearch）
   ├─ 刷新磁盘缓冲区
   ├─ [倒数 5 秒，可取消]
   │
   ├─ 写入 BCD bootsequence（一次性，用后自动清除）
   ├─ 再次刷盘
   └─ reboot
          │
          ▼
   Boot Manager 读取 bootsequence → 引导 WinPE
          │
          ▼
   WinPE startnet.cmd：wpeinit → wpeutil reboot
          │
          ▼
   Boot Manager 自动清除 bootsequence → 引导回 Windows
```

**BCD 写入方式**：通过 `RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE)` 绕过 `HKLM\BCD00000000` 的 ACL，无需 SYSTEM 权限，无需 bcdedit。

---

## 系统要求

- Windows 10/11 64-bit
- 管理员权限（Administrator，非 SYSTEM）
- Python 3.8+，仅需标准库 + `pyyaml`
- WinPE 或 WinRE 环境（见下方配置）

---

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml
```

### 2. 配置 WinPE（首次使用，每台机器做一次）

**方案 A：用系统自带 WinRE（零配置，推荐先试）**

系统若有 WinRE，程序会自动发现，无需任何配置。验证：

```bash
python hot_restart.py --dry-run
```

日志中出现 `Auto-discovered BCD entries` 且列表非空即可。

**方案 B：添加干净 WinPE（已安装 Windows ADK）**

```bash
# 1. 用 ADK copype 构建 WinPE 工作目录（每台机器做一次，WIM 放在持久盘）
python tools/build_pe.py

# 2. 每次开机后，把干净 WinPE 写入系统 BCD（必须每次会话执行，见下方说明）
python bcd_add_winpe.py
# 或指定已构建的 WIM：
python bcd_add_winpe.py --wim D:\DKTM_PE\media\sources\boot.wim
```

> ⚠️ **每次开机都要重跑 `bcd_add_winpe.py`**
> 系统 BCD 存放在 `C:\Boot\BCD`，而 C: 盘受写过滤保护（kdisk），每次重启后还原为
> 服务器镜像，上次写入的 BCD 条目全部消失。`bcd_add_winpe.py` 是 **per-session 工具**，
> 不是一次性安装步骤。同理，WIM 文件会被复制到 `C:\Recovery\WindowsRE\winpe.wim`，
> 重启后也会消失，由脚本在每次会话开始时重新就位。

完成后，程序自动优先使用干净 WinPE（`{7619dcc9-fafe-11d9-b411-000476eba25f}`），以 WinRE 为 fallback。

### 3. 使用

**图形界面（推荐）**

```bash
python gui.py
```

**命令行**

```bash
python hot_restart.py          # 正式执行（有 5 秒取消窗口）
python hot_restart.py --force  # 跳过确认直接执行
python hot_restart.py --dry-run  # 模拟，不做任何系统变更
```

---

## 配置文件

`config.yaml`（留空 `winpe_entry_ids` 则自动发现）：

```yaml
executor:
  mode: real-run
  auto_reboot: false
  transition_method: auto   # auto | bcd | winre
  fallback_method: winre
  marker_path: "C:\\dktm_transition.marker"
  winpe_entry_ids: []       # 空 = 运行时从 BCD 自动发现
```

如需固定 GUID，手动填入：

```yaml
  winpe_entry_ids:
    - "{7619dcc9-fafe-11d9-b411-000476eba25f}"  # 干净 WinPE
    - "{300209a8-6279-11e6-90e0-000c295c2276}"  # WinRE（机器相关）
```

---

## 安全机制

| 机制 | 说明 |
|------|------|
| BCD 备份 | 写入前 `BCD → BCD.dktm.bak` |
| 写后稽核 | 验证 bootmgr / displayorder / bootsequence 三个不变量 |
| 稽核失败自动还原 | 从 `.bak` 覆盖回 `BCD`，抛出异常终止流程 |
| 一次性 bootsequence | Boot Manager 使用后自动清除，即使 WinPE 崩溃也不会死循环 |
| GUI 取消零代价 | 倒数阶段 BCD 尚未写入，取消不需要任何回滚 |
| 提交失败自动回滚 | 写 BCD 异常时自动调用 rollback_transition |

---

## 项目结构

```
DKTM/
├── gui.py                  # 图形界面（推荐入口）
├── hot_restart.py          # 命令行入口
├── bcd_add_winpe.py        # ★ per-session BCD 写入工具（受限环境必用）
├── config.yaml             # 配置文件
├── dktm/
│   ├── platform_windows.py # BCD 写入 / 权限 / 服务 / 磁盘刷新
│   ├── platform_posix.py   # POSIX 占位（dry-run 用）
│   ├── platform_ops.py     # 平台抽象层
│   ├── executor.py         # 命令执行器
│   └── config.py           # 配置加载 / 合并
├── docs/
│   ├── cafe_env_analysis.md  # 网咖环境调研（BCD ACL / 权限 / 写过滤机制 / GUID 来源）
│   ├── testing_notes.md      # 实机测试记录与踩坑总结
│   └── WINPE_BUILD_GUIDE.md
└── tools/
    ├── build_pe.py         # WinPE 构建辅助（copype + DISM）
    └── setup_bcd.py        # ⚠️ 仅限 bcdedit 可用的标准环境；受限环境请用 bcd_add_winpe.py
```

---

## 典型使用场景

### 替代「安装后重启」

安装 WSL、Docker Desktop 或其他需要重启才能激活的 Windows Feature 后，安装程序会弹出「需要重启」提示。

在受限环境（如网咖）中，普通重启意味着经历完整的开机流程，可能触发管理软件的干预。用热重启代替：

```
安装 WSL / Docker / Windows Feature
    ↓
不点安装程序的「立即重启」
    ↓
python gui.py  →  HOT RESTART
    ↓
回到 Windows，Feature 正常激活，可直接使用
```

热重启走 Windows → WinPE（秒过）→ Windows，等效于一次完整重启，内核模块和 Feature 会正常初始化，安装在磁盘上的任何数据不受影响。

---

## 常见问题

**bcdedit 报权限错误？**

正常现象。`HKLM\BCD00000000` 的 ACL 只允许 SYSTEM 写入，Administrator 也被拒。
DKTM 通过 `RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE)` + `SeRestorePrivilege` 绕过，无需 bcdedit。

**如何确认 WinPE 会自动返回 Windows？**

干净 WinPE 的 `startnet.cmd` 内容为：
```bat
@echo off
wpeinit
wpeutil reboot
```
wpeinit 完成后立刻重启，Boot Manager 因 bootsequence 已被自动清除，直接引导回 Windows。

**重启后没有进 WinPE？**

运行 `python hot_restart.py --dry-run` 确认日志中有 `Auto-discovered BCD entries` 且非空。
若为空，运行 `python bcd_add_winpe.py` 添加 WinPE 条目，或确认系统有启用的 WinRE（`reagentc /info`）。
在受写过滤保护的网咖环境，**每次开机后都需要重跑 `bcd_add_winpe.py`**，因为 C: 盘重启后还原。

**WIM 文件在哪？各路径的关系是什么？**

| 路径 | 说明 | 持久性 |
|------|------|--------|
| `D:\DKTM_PE\media\sources\boot.wim` | copype 构建产物，持久盘存储 | 跨重启保留 |
| `C:\Recovery\WindowsRE\winpe.wim` | `bcd_add_winpe.py` 每次会话复制至此 | 重启后消失 |
| BCD device 元素中的路径 | `\Recovery\WindowsRE\winpe.wim`（C: 分区内） | 随 C: 一起清空 |

Boot Manager 从 `C:\Recovery\WindowsRE\winpe.wim` 启动，因此每次会话 `bcd_add_winpe.py` 会同时完成两件事：复制 WIM 文件到位 + 写入 BCD 条目。

**网咖环境能用吗？**

可以，DKTM 最初就是为网咖设计的。参见 [`docs/cafe_env_analysis.md`](docs/cafe_env_analysis.md) 了解详细的环境分析和安全性评估。
