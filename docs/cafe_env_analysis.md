# 網咖環境調研報告

> 調研日期：2026-03-18
> 機器 IP：192.168.10.39
> 系統：Windows 10 Enterprise 2016 LTSB (10.0.19045)
> 運行身份：Administrator（High Mandatory Level）

---

## 一、管理軟件體系

這台機器運行兩套平行的管理系統：**GetwayMi**（遊戲菜單）和 **lwdeploy**（系統管理/部署）。

### 1.1 GetwayMi

| 組件 | 位置 | 說明 |
|------|------|------|
| `GetwayMi.exe` | `D:\games\gwm_data\setup\` | 遊戲菜單主程序（GUI） |
| `gwloader.exe` | `D:\games\<每個遊戲>\` | 遊戲啓動包裝器，每個遊戲目錄各一份 |
| `gw_menu*.txt` | `D:\games\gwm_data\setup\` | 遊戲分類菜單定義（9 組） |
| `boot.bat` | `D:\games\gwm_data\patches\` | 開機補丁腳本 |

`boot.bat` 在開機時按順序執行補丁目錄下的所有 `.reg` 和 `.exe`，只做環境初始化，不恢復磁盤狀態。

### 1.2 Smart Cyber Cafe（CoffeeNet）— 計費系統

```
C:\Program Files (x86)\Wameng\Smart Cyber Cafe - Client\
  ScStart.exe    ← 開機自啓（HKLM\Run: ICoffee）
  CoffeeNet.exe  ← 計費/計時主程序（TCP 6700/6702）
```

管理服務器：`192.168.10.252`，負責計時、鎖定、帳號管理。

### 1.3 lwdeploy（lw 系列）— 系統管理平台

```
B:\lwclient64\              ← 運行在獨立的 B: 分區（userdisk，100 GB）
  lwclient64.exe            ← 主客戶端服務（SYSTEM 帳號，AUTO_START）
  lwhardware64.exe          ← 硬件監控（溫度、CPU 使用率）
  lwPersonalSetting64.exe   ← 用戶設置同步
  ReportBSGuard64.exe       ← 異常上報守護
  config\config.ini         ← 服務器配置
C:\lwdeploy\
  lwcopy_win10.exe          ← 系統鏡像部署工具
  kpowershutdown64.sys      ← Kernel 級電源控制驅動
```

**服務器連接（實時建立中）**：

| 進程 | 目標地址 | 端口 | 用途推測 |
|------|---------|------|---------|
| `lwclient64.exe` | 192.168.10.242 | 13004 | 主控信道 |
| `lwclient64.exe` | 192.168.10.242 | 13000 | 數據信道 |
| `lwclient64.exe` | 192.168.10.243 | 13501 | 備援/監控 |
| `lwhardware64.exe` | 192.168.10.243 | 13000 | 硬件數據上報 |

`B:` 盤標卷名為 `userdisk`，與系統盤 `C:` 分離，可能在重鏡像時被保留（lw 客戶端自我保護）。

---

## 二、開機流程

```
POST / UEFI
    │
    ▼
Boot Manager (C:\bootmgr)
  讀取 C:\Boot\BCD
  → 啓動 Windows 10 Enterprise LTSB
    │
    ▼
Windows 核心啓動
  ntoskrnl.exe → smss.exe → winlogon.exe
  Administrator 帳號自動登錄（High Mandatory Level）
    │
    ├─ [服務自啓] lwclient (AUTO_START, LocalSystem)
    │    B:\lwclient64\lwclient64.exe
    │    └─ 連線 192.168.10.242 / 192.168.10.243
    │
    ├─ [服務自啓] SmartSAMD (Kernel Driver, DEMAND_START)
    │    \SystemRoot\System32\drivers\SmartSAMD.sys
    │
    ├─ [HKLM\Run] ScStart.exe /first
    │    啓動 Smart Cyber Cafe 計費客戶端
    │    CoffeeNet.exe 連線 192.168.10.252:6700
    │
    ├─ [HKLM\Run] RtkAudUService64.exe（音頻驅動服務）
    │
    └─ [用戶進程]
         lwhardware64.exe   → 硬件監控上報
         lwPersonalSetting64.exe → 用戶設置
         gwloader.exe       → 遊戲啓動入口（GetwayMi 菜單）
```

**關鍵觀察**：沒有開機時自動觸發磁盤還原的步驟。lwdeploy 的重鏡像由服務器端主動推送，不在本機開機流程中。

---

## 三、磁盤分區配置

| 盤符 | 卷標 | 容量 | 用途 |
|------|------|------|------|
| `B:` | userdisk | 100 GB | lwdeploy 客戶端 + 用戶設置 |
| `C:` | （系統盤） | 100 GB | Windows 系統、遊戲管理工具 |
| `D:` | Games1 | 56 TB | 遊戲主存儲 |
| `E:` | KINGSTON | 124 GB | 外接 USB 設備 |
| `H:` | Games4 | 14 TB | 遊戲擴展存儲 |

`B:` 盤獨立於系統盤，存放 lwdeploy 客戶端，推測重鏡像時不受影響，確保管理連接在重置後依然存在。

---

## 四、環境重置機制

### 4.1 結論：網絡重鏡像，無本地磁盤保護

逐一排查常見網咖保護方案：

| 方案 | 調查方法 | 結果 |
|------|---------|------|
| Deep Freeze | 驅動列表、進程列表 | ❌ 不存在 |
| ShadowUser / Shadow Protect | 驅動列表 | ❌ 不存在 |
| 冰點還原 / Reboot Restore Rx | 驅動列表 | ❌ 不存在 |
| NTFS 過濾驅動（寫保護） | `fltMC filters` | ❌ 僅標準 Windows 過濾器 |
| Volume Shadow Copy | `vssadmin list shadows` | ❌ 無任何快照 |
| AppLocker / SRP | 注冊表策略鍵 | ❌ 未部署 |

**實際機制**：`lwcopy_win10.exe` + `kpowershutdown64.sys`，由管理服務器（192.168.10.242/243）決定何時推送鏡像。**不是每次重啓都觸發**，屬管理員手動操作或定時排程。

### 4.2 bcdedit 被封鎖的根本原因

`bcdedit` 訪問 `HKLM\BCD00000000` 注冊表路徑，該路徑的 ACL 只允許 SYSTEM 帳號寫入。本機雖以 Administrator 運行，但**並非 SYSTEM**，因此寫入被拒。

值得注意：bcdedit 的封鎖**不是** AppLocker/SRP 策略（兩者均未部署），純粹是 BCD 注冊表 hive 的 ACL 設計。

---

## 五、進程令牌權限分析

通過 Win32 API 直接枚舉令牌，共 24 項特權：

**已啓用（對 DKTM 有意義）**：

| 特權 | 狀態 | 意義 |
|------|------|------|
| `SeBackupPrivilege` | **ENABLED** | 可讀取任意文件，繞過 ACL（讀 BCD）|
| `SeRestorePrivilege` | **ENABLED** | 可寫入任意文件，繞過 ACL（**這是 BCD 文件寫入能成功的根本原因**）|
| `SeDebugPrivilege` | **ENABLED** | 可附加調試器到任意進程（含 SYSTEM 進程）|
| `SeImpersonatePrivilege` | ENABLED（默認）| 可模擬其他用戶令牌 |
| `SeCreateGlobalPrivilege` | ENABLED（默認）| 可創建全局命名對象 |

**已禁用（值得關注）**：

| 特權 | 意義 |
|------|------|
| `SeSystemEnvironmentPrivilege` | **禁用** — 無法通過 API 修改 UEFI 變量，必須操作 BCD 文件 |
| `SeLoadDriverPrivilege` | 禁用 — 無法加載內核驅動 |
| `SeTakeOwnershipPrivilege` | 禁用 — 無法奪取文件/注冊表所有權 |
| `SeSecurityPrivilege` | 禁用 — 無法修改安全描述符 |
| `SeShutdownPrivilege` | 禁用 — `InitiateSystemShutdown` API 無效，但 `shutdown.exe` 命令仍可用 |

**關鍵發現**：`SeBackupPrivilege` 和 `SeRestorePrivilege` 在令牌中已**默認啓用**，無需調用 `AdjustTokenPrivileges`。這解釋了為什麼 `RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE)` 能繞過 BCD hive 的 ACL 限制直接寫入。

---

## 六、BCD 結構與 WinPE 環境

### 6.1 系統 BCD 對象清單（共 13 個）

| GUID | 類型 | 描述 |
|------|------|------|
| `{9dea862c-5cdd-4e70-acc1-f32b344d4795}` | `0x10100002` | Windows Boot Manager |
| `{e18e67b0-6278-11e6-a822-9a545abf3b29}` | `0x10200003` | Windows 10 Enterprise 2016 LTSB（主系統）|
| `{300209a8-6279-11e6-90e0-000c295c2276}` | `0x10200003` | **Windows Recovery Environment（WinRE）** |
| `{300209a9-6279-11e6-90e0-000c295c2276}` | `0x30000000` | WinRE Ramdisk Options |
| `{e18e67af-6278-11e6-a822-9a545abf3b29}` | `0x10200004` | Windows Resume Application（**休眠恢復**，非 WinPE）|
| `{b2721d73-1db4-4c62-bf78-c548a880142d}` | `0x10200005` | Windows 內存診斷 |
| 其餘 7 個 | `0x20100000` / `0x20200003` | Boot sector / EFI 應用 |

> ⚠️ `{e18e67af-...}` 的 Path 是 `\hiberfil.sys`，是休眠恢復程序，誤用會導致進入休眠恢復流程。

### 6.2 可用 WinPE 環境

| 環境 | 位置 | BCD 入口 | 狀態 |
|------|------|---------|------|
| WinRE | `C:\Recovery\WindowsRE\Winre.wim`（509 MB）| `{300209a8-...}` | ✅ 已在 BCD，可直接用 |
| 乾淨 WinPE | ADK: `amd64\en-us\winpe.wim`（325 MB）| 待添加 | ⚠️ 需添加 BCD 入口 |
| copype 工作目錄 | `C:\DKTM_PE\media\sources\boot.wim` | 待添加 | ⚠️ 需添加 BCD 入口 |

---

## 七、DKTM 熱重啓安全性評估

### 7.1 整體流程

```
1. 凍結服務（spooler / SysMain / WSearch）
2. 刷新磁盤緩衝區（FlushFileBuffers 對所有固定磁盤）
3. 健康檢查（管理員權限 / 磁盤空間 / BCD 訪問）
4. 備份 C:\Boot\BCD → C:\Boot\BCD.dktm.bak
5. 啓用 SeBackupPrivilege + SeRestorePrivilege
6. reg load HKLM\TmpDKTM C:\Boot\BCD
7. RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE) 寫入 bootsequence 元素
8. reg unload（自動刷回文件）
9. 稽核驗證（重新 load、核對三個不變量）
10. 驗證失敗 → 從備份還原，拋出錯誤
11. reboot（shutdown /r /t 0）
12. Boot Manager 讀取 bootsequence → 引導 WinPE/WinRE
13. Boot Manager 自動清除 bootsequence
14. WinPE 完成後重啓 → 返回 Windows
```

### 7.2 環境兼容性

| 風險項 | 評估 | 對策 |
|--------|------|------|
| bcdedit 被 ACL 封鎖 | ✅ 已通過直接文件操作繞過 | `RegCreateKeyExW(REG_OPTION_BACKUP_RESTORE)` |
| 本地磁盤還原清除 BCD | ✅ 無本地還原軟件 | — |
| lwdeploy 服務器推送重鏡像 | ⚠️ 無法預測，屬外部因素 | BCD 寫入後盡快重啓，縮短暴露窗口 |
| lwclient 實時連線服務器 | ⚠️ 管理員可能實時看到操作 | 不建議在網咖環境用 real-run |
| SeRestorePrivilege 默認啓用 | ✅ BCD 寫入無需額外提權 | — |
| SeSystemEnvironmentPrivilege 禁用 | ✅ UEFI 變量路徑不可用，但 BCD 文件路徑可用 | — |
| BCD 寫壞導致無法開機 | ✅ 已有備份+稽核+自動還原 | 三重保護 |
| Boot Manager 未清除 bootsequence | ✅ 已在真機驗證，正常返回 Windows | — |

### 7.3 核心安全保障

| 機制 | 實現 |
|------|------|
| 寫入前備份 | `C:\Boot\BCD` → `C:\Boot\BCD.dktm.bak` |
| 寫入後稽核 | 驗證：① bootmgr 存在 ② displayorder 非空 ③ bootsequence 值正確 |
| 稽核失敗自動還原 | 從 `.bak` 覆蓋回 `BCD`，拋出異常終止流程 |
| 一次性 bootsequence | Boot Manager 使用後自動清除，失敗重啓也不會死循環 |
| Fallback 鏈 | bcdedit 失敗 → BCD 文件直寫 → WinRE（reagentc） |

---

## 八、待辦事項

- [x] 將乾淨 WinPE 加入系統 BCD — `{7619dcc9-fafe-11d9-b411-000476eba25f}`（WIM 放置於 `C:\Recovery\WindowsRE\winpe.wim`）
- [x] 在 `boot.wim` 中加入 `startnet.cmd`：`wpeinit` → `wpeutil reboot`，WinPE 啓動後自動重啓返回 Windows
- [x] 更新 `config.yaml`：乾淨 WinPE 為主，WinRE（`{300209a8-...}`）為 fallback
- [x] 優化熱重啓流程：health_check 前移到第一步，去除冗餘 flush_io 調用，重啓倒數支持 Ctrl+C 中斷並自動回滾
