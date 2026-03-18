# 網咖環境調研與重置機制評估

> 調研日期：2026-03-18
> 機器：Windows 10 Enterprise 2016 LTSB (10.0.19045)
> 身份：Administrator（High Mandatory Level）

---

## 一、網咖管理軟件

### 主要組件

| 軟件 | 路徑 | 功能 |
|------|------|------|
| **GetwayMi** | `D:\games\gwm_data\setup\GetwayMi.exe` | 遊戲菜單主程序、遊戲啓動管理 |
| **gwloader.exe** | `D:\games\<各遊戲>\gwloader.exe` | 各遊戲的啓動包裝器（每個遊戲目錄各一份） |
| **Smart Cyber Cafe Client** | `C:\Program Files (x86)\Wameng\Smart Cyber Cafe - Client\` | 計費、計時、用戶鎖定 |
| **CoffeeNet.exe** | 同上 | 計費服務主程序，TCP 端口 6700/6702 |

### Smart Cyber Cafe 配置（`CoffeeNet.ini`）

```ini
[OPTIONS]
PROTOCOL=TCP/IP
LPORT=6700
SPORT=6702
Host=192.168.10.252        ; 管理服務器 IP
RestartTimeout=0           ; 重啓超時（未啓用）
IdleShutDown=0             ; 閒置關機（未啓用）
AllowNewMember=0
```

### GetwayMi 數據目錄 (`D:\games\gwm_data\`)

- `setup/gw_menu*.txt` — 遊戲菜單分類定義（多達 9 組）
- `setup/GetwayMi.exe` — 主程序（2018 年版本）
- `patches/boot.bat` — 開機時執行的補丁腳本
- `patches/ini/*.ini` — Junction 點管理、遊戲刷新規則

---

## 二、環境重置機制

### 結論：**使用網絡重鏡像（lwdeploy），不使用本地磁盤還原**

經過完整調查，排除了以下常見網咖磁盤保護方案：

| 方案 | 調查結果 |
|------|---------|
| Deep Freeze | ❌ 無相關驅動或進程 |
| Shadow Protect / ShadowUser | ❌ 無相關驅動或進程 |
| 冰點還原 / Reboot Restore Rx | ❌ 無相關驅動或進程 |
| Windows Steady State | ❌ 不支持此 Windows 版本 |
| Volume Shadow Copy (VSS) | ❌ `vssadmin list shadows` 無任何快照 |
| NTFS 過濾驅動 | ❌ `fltMC filters` 僅有標準 Windows 過濾器 |

### lwdeploy（網絡部署工具）

```
C:\lwdeploy\
  ├── lwcopy_win10.exe      # 主程序（2022-12，5.4 MB）
  ├── kpowershutdown64.sys  # Kernel-level 電源控制驅動
  ├── kpowershut.exe        # 電源控制用戶態程序
  ├── config\config.ini     # serverip=192.168.2.251
  └── drivers\devcon64.exe  # 設備管理工具
```

**工作原理**：連接部署服務器（`192.168.2.251`），由服務器端決定何時推送鏡像。不是每次重啓都觸發，通常是管理員手動或定時重鏡像。

**對 DKTM 的影響**：
BCD 修改在重啓後**會持久保留**，不會被本地還原機制清除。若管理員觸發重鏡像則整台機器恢復出廠，屬於極端情況。

### 開機補丁腳本（`boot.bat`）

每次開機執行，只做環境配置，不恢復磁盤狀態：

```bat
for %%i in (D:\games\gwm_data\patches\8\*.reg) do regedit /s %%i
for %%i in (D:\games\gwm_data\patches\8\*.exe) do start /wait %%i
for %%i in (D:\games\gwm_data\patches\9\*.reg) do regedit /s %%i
for %%i in (D:\games\gwm_data\patches\*.reg) do regedit /s %%i
for %%i in (D:\games\gwm_data\patches\*.exe) do start %%i
```

---

## 三、系統開機環境（BCD）分析

### bcdedit 封鎖原因

`bcdedit /enum {bootmgr}` 返回 exit code 1（Access Denied）。

根本原因：`HKLM\BCD00000000` 的 ACL 只允許 SYSTEM 帳號寫入。儘管當前身份是 Administrators（High Mandatory Level），仍無法直接寫入 BCD 注冊表路徑。

### 解決方案：直接修改 `C:\Boot\BCD` 文件

`C:\Boot\BCD` 是普通 NTFS 文件，Administrators 有讀寫權限。通過以下流程繞過 BCDStore 的 ACL 限制：

1. 啓用 `SeBackupPrivilege` + `SeRestorePrivilege`
2. `reg load HKLM\TmpDKTM C:\Boot\BCD`
3. 用 `RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE)` 寫入 bootsequence 元素（`24000002`）
4. `reg unload HKLM\TmpDKTM`（自動刷回文件）

Boot Manager 在 POST 階段直接讀取 `C:\Boot\BCD` 文件，完全繞開 Windows 注冊表 ACL。

### BCD 對象完整清單（共 13 個）

| GUID | 類型 | 描述 |
|------|------|------|
| `{9dea862c-5cdd-4e70-acc1-f32b344d4795}` | 0x10100002 | Windows Boot Manager |
| `{e18e67b0-6278-11e6-a822-9a545abf3b29}` | 0x10200003 | Windows 10 Enterprise 2016 LTSB（主系統） |
| `{300209a8-6279-11e6-90e0-000c295c2276}` | 0x10200003 | **Windows Recovery Environment（WinRE）** |
| `{300209a9-6279-11e6-90e0-000c295c2276}` | 0x30000000 | WinRE Ramdisk Options |
| `{e18e67af-6278-11e6-a822-9a545abf3b29}` | 0x10200004 | Windows Resume Application（休眠恢復，非 WinPE） |
| `{b2721d73-1db4-4c62-bf78-c548a880142d}` | 0x10200005 | Windows 內存診斷 |
| 其餘 7 個 | 0x20100000 / 0x20200003 | Boot sector / EFI 應用 |

### WinRE Ramdisk 配置

```
Ramdisk Options {300209a9-...}:
  Image Path : \Recovery\WindowsRE\boot.sdi (SDI 文件)
  WIM Device : locate 類型（指向系統分區上的 Winre.wim）

WinRE Entry {300209a8-...}:
  Device     : ramdisk → {300209a9-...}
  Path       : \windows
  Description: Windows Recovery Environment
```

---

## 四、WinPE 環境評估

### 現有環境

#### WinRE（可立即使用）

```
C:\Recovery\WindowsRE\Winre.wim    509 MB
BCD 入口: {300209a8-6279-11e6-90e0-000c295c2276}
```

- 優點：已配置在 BCD 中，無需額外設置
- 缺點：進入後顯示 Windows 恢復環境菜單，需用戶交互

#### Windows ADK WinPE Add-on（已安裝）

```
C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\
  Windows Preinstallation Environment\
    amd64\en-us\winpe.wim    325 MB  ← 乾淨的 WinPE 基礎鏡像
    copype.cmd                        ← 構建工作目錄工具
    MakeWinPEMedia.cmd                ← 製作開機媒體工具
```

#### copype 生成的工作目錄（已生成）

```
C:\DKTM_PE\
  media\
    Boot\BCD             ← copype 自帶 BCD（ramdisk 引導）
    Boot\boot.sdi        ← 開機 SDI 文件
    sources\boot.wim     ← WinPE 鏡像（325 MB）
    bootmgr / bootmgr.efi
```

copype BCD 中 WinPE 入口 GUID：`{7619dcc9-fafe-11d9-b411-000476eba25f}`
對應 Ramdisk Options GUID：`{7619dcc8-fafe-11d9-b411-000476eba25f}`

### 注意事項

**`{e18e67af-...}` 不是 WinPE**，是 `Windows Resume Application`（`\hiberfil.sys`），指向休眠恢復程序。誤用此 GUID 作為 bootsequence 目標會進入休眠恢復流程而非 WinPE。

---

## 五、DKTM 熱重啓評估

### 熱重啓流程

```
freeze_services → flush_buffers → health_check
→ handover_control → commit_transition（寫 bootsequence）
→ reboot → Boot Manager 讀取 bootsequence → 進入 WinPE/WinRE
→ Boot Manager 自動清除 bootsequence → WinPE 完成後重啓 → 回 Windows
```

bootsequence（BCD 元素 `24000002`）是一次性的：Boot Manager 使用後自動清除，下次重啓回到正常 Windows，**無需 WinPE 端做任何額外操作**。

### 當前配置（`config.yaml`）

```yaml
executor:
  mode: real-run
  transition_method: auto     # bcdedit → BCD 文件直寫 → WinRE 三重 fallback
  winpe_entry_ids:
    - "{300209a8-6279-11e6-90e0-000c295c2276}"   # WinRE
  marker_path: "C:\\dktm_transition.marker"
```

### 安全機制

| 機制 | 實現方式 |
|------|---------|
| 寫入前備份 | `C:\Boot\BCD` → `C:\Boot\BCD.dktm.bak` |
| 寫入後稽核 | 驗證 bootmgr 存在、displayorder 非空、bootsequence 值正確 |
| 驗證失敗自動還原 | 從 .bak 覆蓋回 BCD，並拋出錯誤 |
| 回滾後驗證 | 確認 bootsequence 元素已消失 |

### 網咖環境兼容性

| 項目 | 結論 |
|------|------|
| bcdedit 封鎖 | ✅ 已通過 BCD 文件直寫繞過 |
| 磁盤還原軟件 | ✅ 本機無磁盤還原，BCD 修改持久生效 |
| 管理員權限 | ✅ High Mandatory Level，足夠操作 |
| Boot Manager 清除 bootsequence | ✅ 已在測試機驗證，正常返回 Windows |
| lwdeploy 重鏡像風險 | ⚠️ 管理員手動觸發時整機恢復，屬不可控外部因素 |

---

## 六、待辦事項

- [ ] 將乾淨 WinPE（`{7619dcc9-...}`）加入系統 BCD，作為比 WinRE 更優先的熱重啓目標
- [ ] 在 `boot.wim` 中加入 `startnet.cmd` 自動重啓腳本（`wpeutil reboot`）
- [ ] 更新 `config.yaml`：WinPE 為主，WinRE 為 fallback
- [ ] 優化熱重啓流程（health_check 前移、合并冗餘步驟、縮短等待時間）
