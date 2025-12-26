# WinPE 构建指南 - DKTM 集成

本指南详细说明如何构建自定义 WinPE 镜像，并将其集成到 DKTM（Dynamic Kernel Transition Mechanism）系统中。

---

## 📋 目录

1. [前提条件](#前提条件)
2. [安装 Windows ADK](#安装-windows-adk)
3. [创建基础 WinPE 镜像](#创建基础-winpe-镜像)
4. [自定义 WinPE（集成 DKTM）](#自定义-winpe集成-dktm)
5. [创建 BCD 启动条目](#创建-bcd-启动条目)
6. [测试与验证](#测试与验证)
7. [故障排除](#故障排除)

---

## 前提条件

### 系统要求

- **操作系统**: Windows 10/11 (64-bit)
- **权限**: 管理员权限
- **磁盘空间**: 至少 10 GB 可用空间
- **内存**: 建议 8 GB 或更多

### 安全设置

在开始之前，**必须**暂时禁用以下功能：

```powershell
# 检查 BitLocker 状态
manage-bde -status

# 如果启用了 BitLocker，需要暂停保护
manage-bde -protectors -disable C:

# 检查 Secure Boot 状态（需要重启进入 BIOS/UEFI 禁用）
Confirm-SecureBootUEFI
```

⚠️ **重要**: 完成配置后记得重新启用这些安全功能！

---

## 安装 Windows ADK

### 1. 下载 Windows ADK

访问 Microsoft 官方页面下载最新版本：

- **ADK 主程序**: [Windows ADK 下载](https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install)
- **WinPE 插件**: 必须单独下载（与 ADK 版本匹配）

**当前推荐版本**（2025）:
- Windows ADK 10.1.28000.1 (November 2025)
- Windows PE add-on for ADK

### 2. 安装 ADK

```powershell
# 运行 ADK 安装程序
# 至少选择以下组件：
# ✓ Deployment Tools
# ✓ Windows Preinstallation Environment (Windows PE)
# ✓ Imaging and Configuration Designer (ICD)

# 默认安装路径：
# C:\Program Files (x86)\Windows Kits\10\
```

### 3. 安装 WinPE 插件

```powershell
# 安装完 ADK 后，立即安装 WinPE add-on
# 使用相同的安装路径
```

---

## 创建基础 WinPE 镜像

### 1. 启动部署工具环境

```powershell
# 以管理员身份运行
# 开始菜单 → Windows Kits → Deployment and Imaging Tools Environment
```

### 2. 创建工作目录

```powershell
# 设置架构（amd64 = 64位，x86 = 32位）
$ARCH = "amd64"

# 创建 WinPE 工作文件
copype $ARCH C:\WinPE_DKTM
```

**目录结构**:
```
C:\WinPE_DKTM\
├── fwfiles\        # UEFI 启动文件
├── media\          # ISO 源文件
│   ├── Boot\
│   ├── EFI\
│   └── sources\
│       └── boot.wim    # 核心镜像
└── mount\          # 挂载点（稍后创建）
```

### 3. 创建挂载目录

```powershell
New-Item -Type Directory -Path C:\WinPE_DKTM\mount
```

---

## 自定义 WinPE（集成 DKTM）

### 1. 挂载 WinPE 镜像

```powershell
# 挂载 boot.wim 进行编辑
Dism /Mount-Image `
  /ImageFile:"C:\WinPE_DKTM\media\sources\boot.wim" `
  /Index:1 `
  /MountDir:"C:\WinPE_DKTM\mount"
```

### 2. 添加 WinPE 可选组件

```powershell
# 设置组件路径
$OCs = "C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\Windows Preinstallation Environment\$ARCH\WinPE_OCs"

# 添加必要组件
# WMI 支持（用于系统查询）
Dism /Image:"C:\WinPE_DKTM\mount" `
  /Add-Package `
  /PackagePath:"$OCs\WinPE-WMI.cab"

# PowerShell 支持（如果 DKTM 需要）
Dism /Image:"C:\WinPE_DKTM\mount" `
  /Add-Package `
  /PackagePath:"$OCs\WinPE-NetFx.cab"

Dism /Image:"C:\WinPE_DKTM\mount" `
  /Add-Package `
  /PackagePath:"$OCs\WinPE-Scripting.cab"

Dism /Image:"C:\WinPE_DKTM\mount" `
  /Add-Package `
  /PackagePath:"$OCs\WinPE-PowerShell.cab"

# 网络支持（如果需要远程通信）
Dism /Image:"C:\WinPE_DKTM\mount" `
  /Add-Package `
  /PackagePath:"$OCs\WinPE-Dot3Svc.cab"
```

### 3. 添加 DKTM 执行器

```powershell
# 创建 DKTM 目录
New-Item -Type Directory -Path "C:\WinPE_DKTM\mount\DKTM"

# 复制 DKTM Python 包
Copy-Item -Recurse "C:\path\to\DKTM\dktm" `
  -Destination "C:\WinPE_DKTM\mount\DKTM\"

# 如果需要 Python 运行时
# 下载 Windows Embedded Python 并复制到:
# C:\WinPE_DKTM\mount\Python3
```

### 4. 配置自动启动脚本

创建 `C:\WinPE_DKTM\mount\Windows\System32\startnet.cmd`:

```batch
@echo off
echo ========================================
echo   DKTM WinPE Environment
echo   Dynamic Kernel Transition Mechanism
echo ========================================
echo.

wpeinit

REM 设置网络（如果需要）
REM netsh interface ip set address "Ethernet" static 192.168.1.100 255.255.255.0 192.168.1.1

REM 检查 DKTM 标记文件
if exist X:\DKTM\dktm_transition.marker (
    echo [DKTM] Transition marker detected
    echo [DKTM] Loading transition context...

    REM 执行 DKTM 恢复流程
    cd /d X:\DKTM
    python dktm\dktm.py --mode real-run --rollback

    echo.
    echo [DKTM] Transition complete
    pause
) else (
    echo [DKTM] No transition marker found
    echo [DKTM] Entering maintenance mode
)

REM 启动命令提示符
cmd
```

### 5. 卸载并提交更改

```powershell
# 卸载镜像（保存所有更改）
Dism /Unmount-Image `
  /MountDir:"C:\WinPE_DKTM\mount" `
  /Commit
```

### 6. 生成 ISO 镜像

```powershell
# 创建可启动 ISO
MakeWinPEMedia /ISO C:\WinPE_DKTM C:\WinPE_DKTM.iso

# 或创建可启动 USB（将 U 盘指定为 F:）
# MakeWinPEMedia /UFD C:\WinPE_DKTM F:
```

---

## 创建 BCD 启动条目

### 1. 将 WinPE 部署到系统分区

```powershell
# 创建 WinPE 目录
New-Item -Type Directory -Path "C:\WinPE"

# 复制 WinPE 文件
Copy-Item "C:\WinPE_DKTM\media\*" -Destination "C:\WinPE\" -Recurse
```

### 2. 创建 BCD 条目

```powershell
# 以管理员身份运行

# 1. 创建新的 WinPE 启动条目
$newEntry = bcdedit /create /d "DKTM WinPE" /application osloader

# 从输出中提取 GUID（格式：{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}）
# 假设为 {12345678-1234-1234-1234-123456789abc}

$GUID = "{12345678-1234-1234-1234-123456789abc}"

# 2. 配置 WinPE 条目
bcdedit /set $GUID device "ramdisk=[C:]\WinPE\sources\boot.wim,{ramdiskoptions}"
bcdedit /set $GUID osdevice "ramdisk=[C:]\WinPE\sources\boot.wim,{ramdiskoptions}"
bcdedit /set $GUID path "\Windows\System32\boot\winload.efi"
bcdedit /set $GUID systemroot "\Windows"
bcdedit /set $GUID winpe yes
bcdedit /set $GUID detecthal yes

# 3. 配置 ramdisk 选项
bcdedit /set {ramdiskoptions} ramdisksdidevice partition=C:
bcdedit /set {ramdiskoptions} ramdisksdipath \WinPE\boot\boot.sdi

# 4. 添加到显示顺序（可选）
bcdedit /displayorder $GUID /addlast
```

### 3. 验证配置

```powershell
# 查看所有启动条目
bcdedit /enum all

# 查看特定条目
bcdedit /enum $GUID
```

### 4. 在 DKTM 配置中使用

编辑 `dktm_config.yaml`:

```yaml
executor:
  mode: dry-run  # 测试时使用，实际运行改为 real-run
  auto_reboot: false
  winpe_entry_ids:
    - "{12345678-1234-1234-1234-123456789abc}"  # 你的 WinPE GUID
  marker_path: "C:\\DKTM\\dktm_transition.marker"
```

---

## 测试与验证

### 测试流程

#### 1. Dry-Run 测试

```powershell
# 测试 DKTM 而不实际修改 BCD
python dktm\dktm.py --config dktm_config.yaml --mode dry-run
```

**预期输出**:
```
[INFO] === Committing DKTM Transition ===
[INFO] Target WinPE entry: {12345678-1234-1234-1234-123456789abc}
[DRY-RUN] Would execute: bcdedit /bootsequence {12345678-1234-1234-1234-123456789abc}
[DRY-RUN] Would write marker to C:\DKTM\dktm_transition.marker
[INFO] === Transition Committed ===
```

#### 2. 验证 BCD 设置（非 Dry-Run）

```powershell
# 实际提交过渡（不自动重启）
python dktm\dktm.py --config dktm_config.yaml --mode real-run

# 检查 bootsequence
bcdedit /enum {bootmgr}
```

**预期显示**:
```
bootsequence    {12345678-1234-1234-1234-123456789abc}
```

#### 3. 测试回滚

```powershell
# 回滚过渡
python dktm\dktm.py --rollback

# 验证 bootsequence 已清除
bcdedit /enum {bootmgr}
```

#### 4. 完整过渡测试（虚拟机推荐）

```powershell
# 提交并自动重启
python dktm\dktm.py --config dktm_config.yaml --mode real-run --auto-reboot
```

**预期行为**:
1. 系统重启
2. 进入 DKTM WinPE 环境
3. 执行 `startnet.cmd`
4. 显示 DKTM 状态信息

---

## 故障排除

### 问题 1: bcdedit 命令失败

**症状**: `拒绝访问` 或 `权限不足`

**解决方案**:
```powershell
# 1. 确认管理员权限
whoami /groups | findstr "Administrators"

# 2. 禁用 BitLocker（如果启用）
manage-bde -protectors -disable C:

# 3. 禁用 Secure Boot（重启进入 BIOS/UEFI）
```

### 问题 2: WinPE 启动后黑屏

**症状**: 进入 WinPE 后无响应或黑屏

**解决方案**:
```powershell
# 检查 startnet.cmd 语法
# 确保文件编码为 ANSI（非 UTF-8 with BOM）

# 重新挂载镜像检查
Dism /Mount-Image /ImageFile:"C:\WinPE\sources\boot.wim" /Index:1 /MountDir:"C:\Mount"
type C:\Mount\Windows\System32\startnet.cmd
Dism /Unmount-Image /MountDir:"C:\Mount" /Discard
```

### 问题 3: Python 在 WinPE 中无法运行

**症状**: `python 不是内部或外部命令`

**解决方案**:
```powershell
# 使用 Windows Embeddable Python
# 下载: https://www.python.org/downloads/windows/
# 选择 "Windows embeddable package (64-bit)"

# 解压到 WinPE mount\Python3
# 在 startnet.cmd 中设置 PATH:
set PATH=X:\Python3;%PATH%
```

### 问题 4: bootsequence 未生效

**症状**: 重启后仍进入正常 Windows

**解决方案**:
```powershell
# 1. 检查 bootsequence 是否设置
bcdedit /enum {bootmgr}

# 2. 检查 GUID 是否正确
bcdedit /enum $GUID

# 3. 手动设置一次性启动
bcdedit /bootsequence $GUID

# 4. 如果使用 UEFI，确保:
bcdedit /set $GUID path "\Windows\System32\boot\winload.efi"
```

### 问题 5: Ramdisk 加载失败

**症状**: `The boot configuration data store could not be opened`

**解决方案**:
```powershell
# 检查 boot.sdi 路径
dir C:\WinPE\boot\boot.sdi

# 重新配置 ramdisk
bcdedit /set {ramdiskoptions} ramdisksdidevice partition=C:
bcdedit /set {ramdiskoptions} ramdisksdipath \WinPE\boot\boot.sdi
```

---

## 安全最佳实践

### 完成后重新启用安全功能

```powershell
# 1. 重新启用 BitLocker
manage-bde -protectors -enable C:

# 2. 重新启用 Secure Boot
# 重启进入 BIOS/UEFI 设置

# 3. 验证
Get-BitLockerVolume -MountPoint C:
Confirm-SecureBootUEFI
```

### 备份 BCD

```powershell
# 导出当前 BCD 配置
bcdedit /export C:\BCD_Backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').bcd

# 恢复（如果需要）
bcdedit /import C:\BCD_Backup_20250101_120000.bcd
```

---

## 参考资源

### 官方文档
- [Windows ADK 下载](https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install)
- [WinPE 创建指南](https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/winpe-create-usb-bootable-drive)
- [BCDEdit 参考](https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/bcdedit--bootsequence)

### DKTM 相关
- `docs/ARCHITECTURE.md` - DKTM 架构说明
- `docs/CONFIG.md` - 配置文件指南
- `dktm/platform_windows.py` - Windows 平台实现

---

## 附录：快速参考

### 常用命令

```powershell
# 查看所有启动条目
bcdedit /enum all

# 设置一次性启动
bcdedit /bootsequence {GUID}

# 清除 bootsequence
bcdedit /deletevalue {bootmgr} bootsequence

# 查看 ramdisk 配置
bcdedit /enum {ramdiskoptions}

# 导出 BCD
bcdedit /export C:\bcd_backup.bcd

# 导入 BCD
bcdedit /import C:\bcd_backup.bcd
```

### DKTM 命令

```powershell
# 生成默认配置
python dktm\dktm.py --dump-default-config dktm_config.yaml

# Dry-run 模式测试
python dktm\dktm.py --mode dry-run

# 提交过渡
python dktm\dktm.py --mode real-run

# 回滚过渡
python dktm\dktm.py --rollback
```

---

**文档版本**: 1.0
**最后更新**: 2025-12-26
**适用于**: DKTM v1.0, Windows 10/11, ADK 10.1.28000.1+
