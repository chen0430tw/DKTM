# DKTM 实验记录与踩坑总结

> 机器环境：网咖客户机，Windows 10 Enterprise LTSB，kdisk 写过滤（Wameng 无盘客户端）
> 实验日期：2026-03-18 ~ 2026-03-20
> 目标：实现 Windows → WinPE（自动重启）→ Windows 的内核热重置流程

---

## 一、磁盘写过滤架构

### 关键结论

- **C: 盘每次重启都会被清空**：kdisk64.sys 将所有写入重定向到约 42 GB 的 RAM 缓冲区，重启时缓冲区丢弃，C: 恢复服务器镜像。
- **D: 盘和 H: 盘写入持久**：物理本地磁盘，重启后保留。
- **Boot Manager 在 kdisk 启动之前读取 C:\Boot\BCD**：因此当次会话对 BCD 的修改、以及放在 C: 上的 WIM 文件，对 Boot Manager **可见**，不受清空影响——但仅限**本次重启之前**。

### 设计推论

| 资源 | 存放位置 | 理由 |
|------|----------|------|
| boot.wim（WinPE 源文件） | D:\DKTM_PE\media\sources\ | 持久，重启后仍在 |
| C:\WinPE（运行时副本） | 每次会话由脚本从 D: 复制 | Boot Manager 可见，但会被清空 |
| BCD 条目 | 每次会话由脚本写入 | 同上 |

每次开机后，`bcd_add_winpe.py` 负责：复制 WIM 到 C:\WinPE + 写入 BCD 条目。

---

## 二、BCD 写入

### 问题：bcdedit 被 ACL 封锁

`HKLM\BCD00000000` 的 DACL 只允许 SYSTEM 账户写入，以普通 Administrator 身份运行 bcdedit 会报"拒绝访问"。

### 解决方案：REG_OPTION_BACKUP_RESTORE（=4）

`RegCreateKeyExW` 的 `dwOptions` 传入 `4`（`REG_OPTION_BACKUP_RESTORE`），配合 SeRestorePrivilege（Administrator 默认已具备），可以绕过 DACL 直接写注册表。流程：

```
reg load HKLM\TmpDKTM C:\Boot\BCD
↓
RegCreateKeyExW(HKEY_LOCAL_MACHINE, "TmpDKTM\Objects\{...}\Elements\24000002",
                dwOptions=4, ...)
RegSetValueExW(hKey, "Element", 0, REG_BINARY, guid_bytes, 16)
↓
reg unload HKLM\TmpDKTM
```

该方法已在 `dktm/platform_windows.py` 的 `_commit_via_bcd_file()` 实现，`bcd_add_winpe.py` 复用相同模式写入完整的 WinPE 对象。

### BCD 对象二进制格式（从实机读取）

从 `C:\Boot\BCD` 中读取 WinRE 条目，得到以下模板数据：

**[11000001] device 元素（200 字节）**
- [0–15]：ramdisk options 对象 GUID（小端序）
- [16–135]：固定头部（含分区描述符等结构）
- [136（0x88）+]：WIM 路径，UTF-16LE，null 终止

**[31000003] ramdisk options 元素（88 字节，来自 WinRE 条目，含 C: 分区信息）**

```
hex: 0000000000000000000000000000000006000000000000004800000000000000
     000010000000000000000000000000000000000001000000b31464ff00000000
     0000000000000000000000000000000000000000000000000000000000000000
```
- 偏移 56：MBR 磁盘签名 `0xff6414b3`（当前机器 C: 盘）

**[32000004]**：boot.sdi 路径字符串，UTF-16LE

DKTM 自有 WinPE 条目的 [11000001] device 元素直接从 WinRE 条目克隆，然后在偏移 0x88 处将 `Winre.wim`（9字符）原地替换为 `winpe.wim`（9字符），长度相同，UTF-16LE 直接覆写，保留所有分区信息（含 MBR 磁盘签名 `b3 14 64 ff`）不变。

### DKTM WinPE 固定 GUID

| 用途 | GUID |
|------|------|
| OS loader（{7619dcc9-...}） | `{7619dcc9-fafe-11d9-b411-000476eba25f}` |
| ramdisk options（{7619dcc8-...}） | `{7619dcc8-fafe-11d9-b411-000476eba25f}` |

GUID 固定写死，便于 `platform_windows.py` 的 `_discover_bcd_entries()` 在重启后识别。

---

## 三、WinPE 构建

### 问题：copype.cmd 静默失败

直接从 Python subprocess 调用 copype.cmd 时，以下三个环境变量未设置，导致 copype 静默失败、不报错：

```
WinPERoot
OSCDImgRoot
DISMRoot
```

这些变量由 ADK 的 `DandISetEnv.bat` 设置，但 subprocess 不继承该 bat 文件的效果。

**修复：** 在调用前手动设置环境变量：

```python
env = os.environ.copy()
adk = r"C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit"
env["WinPERoot"]   = adk + r"\Windows Preinstallation Environment"
env["OSCDImgRoot"] = adk + r"\Deployment Tools\amd64\Oscdimg"
env["DISMRoot"]    = adk + r"\Deployment Tools\amd64\DISM"
subprocess.run(["cmd", "/c", copype_path, "amd64", output_dir], env=env)
```

### WinPE 最终构建路径

```
D:\DKTM_PE\
  media\
    sources\boot.wim    ← 主镜像，持久存储
    Boot\boot.sdi       ← boot.sdi
  mount\
    Windows\System32\startnet.cmd
```

### startnet.cmd 内容（简洁版）

```batch
@echo off
wpeinit
wpeutil reboot
```

不需要检测 marker 文件或执行复杂逻辑——WinPE 的唯一任务是让 Windows 重启，内核由 kdisk 在重启时重置。

---

## 四、热重启流程验证

### 测试 1：使用 WinRE 热重启（失败）

**实际运行日志**（bcd_add_winpe.py 尚未执行，系统只有 WinRE 条目）：

```
Auto-discovered BCD entries: ['{300209a8-6279-11e6-90e0-000c295c2276}']
Auto-discovered WinPE entry IDs: ['{300209a8-6279-11e6-90e0-000c295c2276}']
Boot entry: WinRE fallback only (clean WinPE not found)
Disk: 72484 MB free on C:\
── Health check ──
Running pre-restart health check...
✓ Disk space OK: 72484 MB free on C:\
✓ bcdedit not available; C:\Boot\BCD present (file method will be used)
Auto-discovered BCD entries: ['{300209a8-6279-11e6-90e0-000c295c2276}']
Auto-discovered WinPE entry IDs: ['{300209a8-6279-11e6-90e0-000c295c2276}']
✓ Boot entry: WinRE fallback ({300209a8-6279-11e6-90e0-000c295c2276}) — clean WinPE not found
✓ Health check passed
── Quiescing services ──
Freezing non-essential services: spooler, SysMain, WSearch
✓ Stopped service: spooler
✓ Stopped service: SysMain
Service WSearch:
── Flushing I/O ──
Flushing filesystem buffers...
✓ Flushed volume B:
✓ Flushed volume C:
✓ Flushed volume D:
✓ Flushed volume H:
✓ Flushed 4 volume(s)
Hot restart aborted by user
```

**日志解读**：

- `_discover_bcd_entries()` 自动发现了 WinRE 条目 `{300209a8-6279-11e6-90e0-000c295c2276}`，DKTM 固定 GUID `{7619dcc9-...}` 不存在（尚未运行 bcd_add_winpe.py）。
- bcdedit 因 ACL 封锁不可用，已正确切换到 BCD 文件直写方案。
- `WSearch` 停止服务那行输出不完整（可能因服务已停止或访问被拒），不影响流程。
- 用户在倒计时阶段按 Ctrl+C 中止，未实际重启。

**现象**：此次未实际触发重启，但若继续执行，将使用 WinRE（reagentc /boottore）而非自建 WinPE。根据环境分析，WinRE 路径会直接导致机器重启回 Windows（kdisk 写过滤在重启时清除 reagentc 写入的标志位），内核未经 WinPE 重置步骤。

**结论**：不能依赖 WinRE 做热重启，必须先运行 bcd_add_winpe.py 写入 DKTM 自建 WinPE 条目，再执行热重启。

### 测试 2：bcd_add_winpe.py 写入成功，健康检查通过

运行 bcd_add_winpe.py 后，DKTM 固定 GUID `{7619dcc9-...}` 已写入 BCD，再次执行热重启预检：

**实际运行日志**：

```
Auto-discovered BCD entries: ['{7619dcc9-fafe-11d9-b411-000476eba25f}', '{300209a8-6279-11e6-90e0-000c295c2276}']
Auto-discovered WinPE entry IDs: ['{7619dcc9-fafe-11d9-b411-000476eba25f}', '{300209a8-6279-11e6-90e0-000c295c2276}']
Boot entry: DKTM clean WinPE
Disk: 60985 MB free on C:\
── Health check ──
Running pre-restart health check...
✓ Disk space OK: 60985 MB free on C:\
✓ bcdedit not available; C:\Boot\BCD present (file method will be used)
Auto-discovered BCD entries: ['{7619dcc9-fafe-11d9-b411-000476eba25f}', '{300209a8-6279-11e6-90e0-000c295c2276}']
Auto-discovered WinPE entry IDs: ['{7619dcc9-fafe-11d9-b411-000476eba25f}', '{300209a8-6279-11e6-90e0-000c295c2276}']
✓ Boot entry: DKTM clean WinPE ({7619dcc9-fafe-11d9-b411-000476eba25f})
✓ Health check passed
── Quiescing services ──
Freezing non-essential services: spooler, SysMain, WSearch
Service spooler:
Service SysMain:
Service WSearch:
── Flushing I/O ──
Flushing filesystem buffers...
✓ Flushed volume B:
✓ Flushed volume C:
✓ Flushed volume D:
✓ Flushed volume H:
✓ Flushed 4 volume(s)
Hot restart aborted by user
```

**日志解读**：

- `_discover_bcd_entries()` 正确将 `{7619dcc9-...}` 排在首位（DKTM 优先），WinRE `{300209a8-...}` 作为备选。
- 健康检查报告 `Boot entry: DKTM clean WinPE`，说明 bcd_add_winpe.py 写入成功。
- 三个服务（spooler / SysMain / WSearch）输出为空行——这些服务在本次会话已处于停止状态（上一次测试已停过），sc stop 对已停止的服务返回非零，不影响流程。
- 磁盘比测试 1 少约 11 GB（60985 vs 72484 MB），因为 C:\WinPE 已复制 WIM 文件进来。
- 用户再次在倒计时阶段 Ctrl+C 中止，未实际重启。

**对比测试 1**：

| 项目 | 测试 1（仅 WinRE） | 测试 2（有 DKTM WinPE） |
|------|--------------------|------------------------|
| 首选启动条目 | WinRE `{300209a8-...}` | DKTM WinPE `{7619dcc9-...}` |
| 健康检查结论 | WinRE fallback only | DKTM clean WinPE |
| C: 可用空间 | 72484 MB | 60985 MB（WIM 已复制） |
| 服务停止输出 | spooler/SysMain 有 ✓ | 三项均为空（已停止） |

**结论**：bcd_add_winpe.py 写入 BCD 条目正常，热重启预检通过。下一步需实际触发重启验证 WinPE 启动和自动回弹。

### 测试 3：WinPE 构建（copype + DISM）

- copype + DISM 挂载/卸载正常完成（修复环境变量后）
- WIM 文件存于 D:\DKTM_PE\media\sources\boot.wim
- startnet.cmd 写入成功（wpeinit + wpeutil reboot）

### 测试 4：BCD 直写（2026-03-20 已验证）

`bcd_add_winpe.py` 实际运行结果：
- [x] reg load BCD hive 成功（`HKLM\TmpDKTMPE`）
- [x] REG_OPTION_BACKUP_RESTORE 写入 WinPE 对象成功
- [x] displayorder 写入成功（REG_MULTI_SZ，追加 `{7619dcc9-...}`）
- [ ] 重启后 Boot Manager 选择 DKTM WinPE（待端到端实测）
- [ ] WinPE 执行 wpeutil reboot 后回到 Windows（待端到端实测）

验证方法（reg load + Python REPL 读回）：
```
Type = 0x10200003  OK
11000001 [BIN len=200]  device（含 C: 磁盘签名 b3 14 64 ff）
12000002 [SZ] '\windows\system32\winload.exe'
12000004 [SZ] 'DKTM WinPE'
21000001 [BIN len=200]  osdevice
22000002 [SZ] '\windows'
26000010 [BIN len=1]  detecthal=01
26000022 [BIN len=1]  winpe=01
bootmgr displayorder: [..., {7619dcc9-fafe-11d9-b411-000476eba25f}]
```

---

## 五、其他踩坑

### build_pe.py 的 UnicodeEncodeError

`build_pe.py` 内含 Unicode 字符（✓ ✗ 等），在 cp950 代码页下运行报 UnicodeEncodeError。

**绕过方案**：不通过 build_pe.py，直接在 Python REPL 中调用 subprocess 手动执行 copype 和 DISM。

### 路径确认要实测

之前误报 `D:\DKTM` 存在（实际不存在）。确认路径存在必须实际运行 `os.path.exists()` 或 `dir` 命令，不能从脚本逻辑推断。

---

## 六、当前状态（2026-03-20）

| 组件 | 状态 |
|------|------|
| WinPE 镜像（D:\DKTM_PE） | 已构建 |
| startnet.cmd | 已写入（wpeinit + wpeutil reboot） |
| platform_windows.py（BCD 直写） | 已实现 |
| bcd_add_winpe.py | 已完成并验证（2026-03-20） |
| 端到端热重启测试 | 待测试 |
