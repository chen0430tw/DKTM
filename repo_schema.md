# DKTM Repository Schema

**Dynamic Kernel Transition Mechanism - Architecture & Design Documentation**

Version: 1.0.0
Last Updated: 2025-12-26

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Layers](#architecture-layers)
3. [Module Dependency Graph](#module-dependency-graph)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [API Documentation](#api-documentation)
6. [File Hierarchy](#file-hierarchy)
7. [Usage Patterns](#usage-patterns)
8. [Extension Points](#extension-points)

---

## Project Overview

### Purpose

DKTM is a **one-click hot restart system** for Windows that enables kernel-level state reset without full system reboot. It leverages intelligent state detection (SOSA algorithm) to safely transition the system through a WinPE environment, providing a "kexec-like" experience on Windows.

### Core Philosophy

```
User clicks button → SOSA detects safety → Auto-switch to PE → Kernel reset → Auto-return to Windows
                     ↑                                          ↑
                  Jerry checks                            Tom is gone
```

**Not a bcdedit wrapper** - Full automation with intelligent decision-making.

### Key Differentiators

| Feature | Traditional Scripts | DKTM |
|---------|-------------------|------|
| Decision Making | Manual or blind execution | SOSA algorithm (intelligent) |
| PE Generation | Manual ADK operations | Automated build pipeline |
| BCD Configuration | Manual bcdedit commands | Programmatic API |
| Recovery | Manual intervention | Automatic self-cleaning |
| Safety | No checks | Multi-layer validation |

---

## Architecture Layers

### Layer 1: User Interface

```
┌─────────────────────────────────────────────────────┐
│  User Commands                                      │
│  ├─ install.py         (One-click installer)        │
│  └─ hot_restart.py     (One-click hot restart)      │
└─────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Command-line argument parsing
- User feedback and progress reporting
- Error handling and user guidance
- Orchestration of lower layers

### Layer 2: Orchestration

```
┌─────────────────────────────────────────────────────┐
│  DKTM Core Orchestrator (dktm/dktm.py)              │
│  ├─ State detection coordination                    │
│  ├─ Plan generation                                 │
│  └─ Execution workflow management                   │
└─────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Integration of SOSA, Retina, Planner, and Executor
- Workflow state management
- Error recovery coordination
- Configuration management

### Layer 3: Intelligence

```
┌───────────────────────┐  ┌──────────────────────────┐
│  SOSA Algorithm       │  │  Retina Probe            │
│  (spark_seed_sosa.py) │  │  (retina_probe.py)       │
│  ├─ Binary-Twin       │  │  ├─ Gradient detection   │
│  ├─ Explore factor    │  │  ├─ E_mean calculation   │
│  └─ State distribution│  │  └─ Hotspot detection    │
└───────────────────────┘  └──────────────────────────┘
              ↓                        ↓
         ┌────────────────────────────────┐
         │  Adapter (adapter.py)          │
         │  - Event aggregation           │
         │  - Binary-Twin computation     │
         └────────────────────────────────┘
```

**Responsibilities:**
- System state monitoring ("Jerry checking if Tom is gone")
- Safety assessment (E_mean < 0.5 threshold)
- Decision support (explore_factor > 0.3)
- Real-time state representation

### Layer 4: Planning & Execution

```
┌─────────────────────────────────────────────────────┐
│  Plan Generation (plan.py)                          │
│  ├─ Binary-Twin analysis                            │
│  ├─ Multi-phase planning (quiesce → flush → commit) │
│  └─ Command sequence generation                     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Executor (executor.py)                             │
│  ├─ Phase-by-phase execution                        │
│  ├─ Dry-run mode support                            │
│  └─ Platform delegation                             │
└─────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Transition plan generation based on state
- Step-by-step execution with validation
- Rollback capability
- Dry-run simulation

### Layer 5: Platform Abstraction

```
┌─────────────────────────────────────────────────────┐
│  Platform Operations (platform_ops.py)              │
│  - Automatic platform detection                     │
│  - Unified interface                                │
└─────────────────────────────────────────────────────┘
         ↓                              ↓
┌─────────────────────┐    ┌──────────────────────────┐
│  Windows Platform   │    │  POSIX Platform          │
│  (platform_windows) │    │  (platform_posix)        │
│  ├─ BCD operations  │    │  ├─ Stub implementations │
│  ├─ Marker files    │    │  ├─ Development support  │
│  └─ Reboot control  │    │  └─ Testing utilities    │
└─────────────────────┘    └──────────────────────────┘
```

**Responsibilities:**
- BCD manipulation (bootsequence, entry creation)
- Marker file management (JSON metadata)
- Administrator privilege checks
- System reboot coordination

### Layer 6: Automation Tools

```
┌──────────────────────┐  ┌──────────────────────────┐
│  WinPE Builder       │  │  BCD Configurator        │
│  (build_pe.py)       │  │  (setup_bcd.py)          │
│  ├─ ADK detection    │  │  ├─ Entry creation       │
│  ├─ copype automation│  │  ├─ Ramdisk config       │
│  ├─ Script injection │  │  ├─ GUID extraction      │
│  └─ Deployment       │  │  └─ Config persistence   │
└──────────────────────┘  └──────────────────────────┘
```

**Responsibilities:**
- Automated WinPE image construction
- DKTM recovery script injection
- BCD entry lifecycle management
- Configuration validation

---

## Module Dependency Graph

### Core Module Dependencies

```
hot_restart.py
    ├─→ dktm.dktm (DKTM orchestrator)
    │   ├─→ dktm.adapter (SOSA adapter)
    │   │   └─→ dktm.spark_seed_sosa (SOSA algorithm)
    │   ├─→ dktm.retina_probe (State detector)
    │   ├─→ dktm.plan (Plan generator)
    │   │   └─→ dktm.command_dict (Command definitions)
    │   ├─→ dktm.executor (Executor)
    │   │   └─→ dktm.platform_ops (Platform abstraction)
    │   │       ├─→ dktm.platform_windows (Windows implementation)
    │   │       └─→ dktm.platform_posix (POSIX stub)
    │   └─→ dktm.config (Configuration management)
    └─→ numpy (External: state matrices)

install.py
    ├─→ tools.build_pe (WinPE builder)
    │   └─→ subprocess (External: ADK tools)
    └─→ tools.setup_bcd (BCD configurator)
        └─→ yaml (External: config persistence)

tools/build_pe.py
    ├─→ pathlib (Standard library)
    ├─→ subprocess (ADK copype, DISM)
    └─→ logging (Progress reporting)

tools/setup_bcd.py
    ├─→ subprocess (bcdedit operations)
    ├─→ yaml (Config read/write)
    └─→ re (GUID extraction)
```

### Circular Dependency Prevention

The architecture explicitly avoids circular dependencies through:

1. **Unidirectional flow**: User → Orchestrator → Intelligence → Platform
2. **Interface segregation**: Platform ops are abstracted
3. **Dependency injection**: Config passed down, not imported up

---

## Data Flow Diagrams

### 1. Installation Flow

```
START: python install.py
    ↓
[Check Prerequisites]
    ├─ Python 3.7+ ✓
    ├─ Administrator ✓
    ├─ numpy, pyyaml ✓
    └─ Windows 10/11 ✓
    ↓
[Create Directories]
    ├─ C:\DKTM
    └─ C:\DKTM\logs
    ↓
[Build WinPE] (tools/build_pe.py)
    ├─ Detect ADK path
    ├─ Run copype.cmd → Generate base PE
    ├─ Mount boot.wim
    ├─ Inject dktm_recovery.cmd
    ├─ Configure startnet.cmd
    ├─ Commit and unmount
    └─ Deploy to C:\WinPE
    ↓
[Setup BCD] (tools/setup_bcd.py)
    ├─ Create BCD entry → Extract GUID
    ├─ Configure ramdisk options
    ├─ Set WinPE boot parameters
    ├─ Validate configuration
    └─ Save to dktm_config.yaml
    ↓
[Verify Installation]
    ├─ Check WinPE files exist
    ├─ Validate dktm_config.yaml
    └─ Verify BCD GUID
    ↓
END: Installation complete ✅
```

### 2. Hot Restart Flow

```
START: python hot_restart.py
    ↓
[Initialize DKTM]
    ├─ Load dktm_config.yaml
    ├─ Initialize SOSA adapter
    ├─ Initialize Retina probe
    └─ Load platform operations
    ↓
[Probe System State] ← "Jerry checking if Tom is gone"
    ├─ Generate state matrix (10x10)
    ├─ Retina detection → E_mean
    ├─ Submit to SOSA adapter
    └─ Compute Binary-Twin + explore_factor
    ↓
[Safety Decision]
    ├─ E_mean < 0.5 ? ────┐
    ├─ explore_factor > 0.3 ?  │
    │  ↓ YES               ↓ NO
    │  [SAFE]         [ABORT]
    │                 "System too busy"
    │                 EXIT
    ↓
[Generate Transition Plan]
    ├─ PHASE 1: quiesce (prepare system)
    ├─ PHASE 2: flush (finalize state)
    └─ PHASE 3: commit (set BCD + reboot)
    ↓
[Execute Plan] (dry-run aware)
    ├─ Execute quiesce commands
    ├─ Execute flush commands
    └─ Execute commit commands
        ├─ Check admin privileges
        ├─ Backup BCD configuration
        ├─ Write marker file (JSON metadata)
        ├─ Set bootsequence → WinPE GUID
        └─ Trigger reboot (if not dry-run)
    ↓
[SYSTEM REBOOTS]
    ↓
[WinPE Environment]
    ├─ Boot into WinPE
    ├─ startnet.cmd auto-executes
    └─ dktm_recovery.cmd runs
        ├─ Clear event logs
        ├─ Reset network stack
        ├─ Flush DNS cache
        ├─ Remove marker file
        ├─ Clear bootsequence ← Auto-cleanup
        └─ wpeutil reboot
    ↓
[SYSTEM REBOOTS AGAIN]
    ↓
END: Back to Windows with fresh kernel ✅
```

### 3. SOSA Decision Flow

```
System State Matrix (10x10)
    ↓
[Retina Probe]
    ├─ Compute gradients (Sobel-like)
    ├─ Generate E_map (edge intensity)
    ├─ Calculate E_mean (average pressure)
    └─ Detect hotspots
    ↓
    └─→ E_mean (float 0-1)
         │
         │
[SOSA Adapter] ←─── Event submission
    ├─ Collect observation vectors
    ├─ Aggregate events (buffer)
    └─ Flush to SOSA algorithm
         ↓
[SparkSeedSOSA]
    ├─ Compute Binary-Twin (continuous + discrete)
    ├─ Calculate explore_factor
    └─ Generate state_distribution (Markov)
         ↓
         └─→ Binary-Twin, explore_factor
              ↓
[Decision Gate]
    if E_mean < 0.5 AND explore_factor > 0.3:
        → SAFE TO RESTART ✅
    else:
        → ABORT, SYSTEM BUSY ⛔
```

---

## API Documentation

### Module: `dktm.platform_ops`

#### Class: `PlatformOperations` (Abstract)

**Purpose**: Unified interface for platform-specific operations.

##### Method: `commit_transition(auto_reboot=False, dry_run=False)`

**Description**: Prepares system to boot into WinPE on next reboot.

**Parameters:**
- `auto_reboot` (bool): If True, immediately triggers reboot
- `dry_run` (bool): If True, logs actions without executing

**Raises:**
- `RuntimeError`: If not running as Administrator (Windows)
- `FileNotFoundError`: If WinPE entry not found in config

**Side Effects:**
- Modifies BCD bootsequence (Windows only)
- Creates marker file at C:\DKTM\dktm.marker.json
- May trigger system reboot

**Example:**
```python
from dktm.platform_ops import get_platform_operations

ops = get_platform_operations(config={
    "executor": {"winpe_entry_ids": ["{guid}"]}
})
ops.commit_transition(auto_reboot=True, dry_run=False)
# System will reboot into WinPE
```

---

##### Method: `reboot(delay_seconds=5, dry_run=False)`

**Description**: Triggers system reboot with countdown.

**Parameters:**
- `delay_seconds` (int): Countdown before reboot
- `dry_run` (bool): If True, logs without rebooting

**Platform Behavior:**
- **Windows**: Uses `shutdown /r /t {delay}`
- **POSIX**: Uses `sudo shutdown -r +{minutes}` (stub)

**Example:**
```python
ops.reboot(delay_seconds=10, dry_run=False)
# System reboots in 10 seconds
```

---

### Module: `dktm.adapter`

#### Class: `SOSAAdapter`

**Purpose**: Bridges DKTM system with SOSA algorithm.

##### Method: `submit_event(obs_vec, event_type)`

**Description**: Submits an observation event for aggregation.

**Parameters:**
- `obs_vec` (np.ndarray): Observation vector (shape: [N,])
- `event_type` (str): Event category (e.g., "probe", "metric")

**Example:**
```python
adapter = SOSAAdapter()
obs = np.random.rand(128)
adapter.submit_event(obs, "probe")
```

---

##### Method: `flush()`

**Description**: Flushes buffer and computes SOSA outputs.

**Returns:**
- `binary_twin` (dict): Continuous and discrete state features
- `explore_factor` (float): Exploration coefficient (0-1)
- `state_distribution` (np.ndarray): Markov state probabilities

**Example:**
```python
bt, explore, dist = adapter.flush()
print(f"Safe to restart: {explore > 0.3}")
```

---

### Module: `dktm.retina_probe`

#### Function: `retina_probe(state_matrix)`

**Description**: Performs gradient-based state detection.

**Parameters:**
- `state_matrix` (np.ndarray): System state matrix (2D)

**Returns:**
- `E_map` (np.ndarray): Edge intensity map
- `E_mean` (float): Average system pressure (0-1)
- `hotspots` (List[Tuple[int, int]]): Coordinates of high-pressure areas

**Algorithm:**
```
1. Apply Sobel-like gradient operators
2. Compute edge magnitude: E = sqrt(Gx^2 + Gy^2)
3. Calculate mean: E_mean = mean(E)
4. Detect hotspots: where E > threshold
```

**Example:**
```python
from dktm.retina_probe import retina_probe
import numpy as np

state = np.random.rand(10, 10)
result = retina_probe(state)
if result["E_mean"] < 0.5:
    print("System is calm, safe to restart")
```

---

### Module: `tools.build_pe`

#### Class: `WinPEBuilder`

##### Method: `build(deploy=False)`

**Description**: Builds WinPE image with DKTM recovery scripts.

**Parameters:**
- `deploy` (bool): If True, deploys to C:\WinPE

**Process:**
1. Detect ADK installation path
2. Run copype.cmd to generate base PE
3. Mount boot.wim
4. Inject dktm_recovery.cmd to startup directory
5. Configure startnet.cmd for auto-execution
6. Commit and unmount image
7. Optionally deploy to system

**Raises:**
- `FileNotFoundError`: If ADK not installed
- `subprocess.CalledProcessError`: If DISM operations fail

**Example:**
```python
builder = WinPEBuilder()
success = builder.build(deploy=True)
# WinPE now available at C:\WinPE
```

---

### Module: `tools.setup_bcd`

#### Class: `BCDConfigurator`

##### Method: `configure(save_config=False)`

**Description**: Creates and configures BCD entry for DKTM WinPE.

**Parameters:**
- `save_config` (bool): If True, saves GUID to dktm_config.yaml

**Process:**
1. Create new BCD entry with `bcdedit /create`
2. Extract GUID from output
3. Configure ramdisk options
4. Set boot parameters (device, path, osdevice)
5. Validate configuration
6. Optionally persist to config file

**Returns:**
- `guid` (str): WinPE entry GUID

**Example:**
```python
configurator = BCDConfigurator()
guid = configurator.configure(save_config=True)
print(f"WinPE entry created: {guid}")
```

---

## File Hierarchy

### Conceptual Organization

```
DKTM/
├─ [USER INTERFACE]
│  ├─ install.py              # Entry point: one-click setup
│  └─ hot_restart.py          # Entry point: one-click restart
│
├─ [CORE PACKAGE]
│  └─ dktm/
│     ├─ [ORCHESTRATION]
│     │  ├─ dktm.py           # Main orchestrator
│     │  └─ config.py         # Configuration management
│     │
│     ├─ [INTELLIGENCE]
│     │  ├─ spark_seed_sosa.py    # SOSA algorithm
│     │  ├─ adapter.py            # SOSA adapter
│     │  └─ retina_probe.py       # State detector
│     │
│     ├─ [PLANNING]
│     │  ├─ plan.py           # Plan generator
│     │  ├─ executor.py       # Plan executor
│     │  └─ command_dict.py   # Command definitions
│     │
│     └─ [PLATFORM]
│        ├─ platform_ops.py       # Abstraction layer
│        ├─ platform_windows.py   # Windows implementation
│        └─ platform_posix.py     # POSIX stub
│
├─ [AUTOMATION TOOLS]
│  └─ tools/
│     ├─ build_pe.py          # WinPE builder
│     └─ setup_bcd.py         # BCD configurator
│
├─ [QUALITY ASSURANCE]
│  └─ check_code.py           # Code quality checker
│
├─ [DOCUMENTATION]
│  ├─ README.md               # User documentation
│  ├─ BUGFIXES.md             # Bug report
│  ├─ repo_index.json         # Structured index
│  ├─ repo_schema.md          # This file
│  └─ docs/
│     └─ WINPE_BUILD_GUIDE.md # Technical guide
│
└─ [CONFIGURATION]
   ├─ requirements.txt        # Python dependencies
   └─ dktm_config.yaml        # Generated during install
```

### File Categories

| Category | Files | Purpose |
|----------|-------|---------|
| **Entry Points** | install.py, hot_restart.py | User-facing commands |
| **Core Logic** | dktm/*.py | Algorithm and orchestration |
| **Automation** | tools/*.py | PE build, BCD setup |
| **Platform** | platform_*.py | OS-specific operations |
| **Quality** | check_code.py | Bug detection |
| **Config** | requirements.txt, dktm_config.yaml | Dependencies and settings |
| **Docs** | *.md, docs/*.md | User and technical guides |

---

## Usage Patterns

### Pattern 1: First-Time Setup

**Scenario**: User wants to install DKTM on fresh Windows system.

**Steps:**
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run one-click installer (as Administrator)
python install.py

# Installer automatically:
# - Creates C:\DKTM directories
# - Builds WinPE with recovery scripts
# - Creates BCD entry
# - Saves configuration to dktm_config.yaml
```

**Expected Output:**
```
>>> Checking prerequisites...
✓ All prerequisites met

>>> Creating directories...
✓ Directories created

>>> Building WinPE...
[build_pe.py output...]
✓ WinPE built and deployed

>>> Setting up BCD...
[setup_bcd.py output...]
✓ BCD configured

>>> Verifying installation...
  WinPE GUID: {12345678-1234-1234-1234-123456789abc}
✓ Installation verified

✅ DKTM Installation Complete!
```

---

### Pattern 2: Safe Testing

**Scenario**: User wants to test hot restart without actually rebooting.

**Steps:**
```bash
# Run with dry-run flag
python hot_restart.py --dry-run
```

**Behavior:**
- Performs full system state detection
- Generates transition plan
- Logs all actions that would be executed
- **Does not modify BCD**
- **Does not trigger reboot**

**Expected Output:**
```
[SOSA] Probing system state...
[RETINA] E_mean = 0.42 (SAFE)
[SOSA] explore_factor = 0.58 (SUFFICIENT)
✓ System state is safe for hot restart

[PLAN] Generated transition plan:
  Phase 1 (quiesce): 3 commands
  Phase 2 (flush): 2 commands
  Phase 3 (commit): 1 command

[DRY-RUN] Would execute: quiesce_network
[DRY-RUN] Would execute: flush_cache
[DRY-RUN] Would set bootsequence: {guid}
[DRY-RUN] Would trigger reboot

✓ Dry-run complete (no changes made)
```

---

### Pattern 3: Production Hot Restart

**Scenario**: User is confident system is ready for actual hot restart.

**Steps:**
```bash
# Execute hot restart (as Administrator)
python hot_restart.py
```

**Behavior:**
- SOSA detects system state
- If safe (E_mean < 0.5, explore_factor > 0.3):
  - Sets BCD bootsequence to WinPE GUID
  - Creates marker file
  - Triggers reboot
- System boots into WinPE
- WinPE executes recovery script
- WinPE clears bootsequence
- System reboots back to Windows

**Timeline:**
```
T+0s:     python hot_restart.py executed
T+2s:     SOSA analysis complete → SAFE
T+5s:     BCD modified, reboot initiated
T+30s:    WinPE boots
T+60s:    Recovery script clears caches, resets network
T+90s:    WinPE clears bootsequence, reboots
T+120s:   Back to Windows with fresh kernel ✅
```

---

### Pattern 4: Forced Execution (Advanced)

**Scenario**: User wants to override safety checks (not recommended).

**Steps:**
```bash
# Force restart even if SOSA deems it unsafe
python hot_restart.py --force
```

**Behavior:**
- Skips E_mean and explore_factor checks
- Directly executes transition plan
- **Warning**: May cause data loss or system instability

**Use Cases:**
- Emergency kernel reset
- Testing SOSA thresholds
- Development debugging

---

### Pattern 5: Custom Configuration

**Scenario**: User has multiple WinPE environments or custom settings.

**Steps:**
```bash
# Use custom configuration file
python hot_restart.py --config my_custom_config.yaml
```

**Configuration Example:**
```yaml
executor:
  winpe_entry_ids:
    - "{custom-winpe-guid}"
  marker_path: "D:\\CustomPath\\marker.json"

sosa:
  e_mean_threshold: 0.4  # Stricter safety
  explore_factor_threshold: 0.5

logging:
  level: DEBUG
  log_dir: "D:\\CustomLogs"
```

---

### Pattern 6: Manual Step-by-Step

**Scenario**: User wants granular control over each installation step.

**Steps:**
```bash
# 1. Build WinPE only
python tools/build_pe.py --deploy

# 2. Configure BCD only
python tools/setup_bcd.py --save-config

# 3. Verify manually
python hot_restart.py --dry-run
```

**Use Cases:**
- Debugging installation issues
- Customizing PE before BCD setup
- Re-running specific steps after failure

---

## Extension Points

### 1. Custom SOSA Algorithms

**Location**: `dktm/spark_seed_sosa.py`

**How to Extend:**
```python
# Create custom SOSA variant
class CustomSOSA(SparkSeedSOSA):
    def _compute_binary_twin(self, aggregated_obs):
        # Your custom Binary-Twin logic
        continuous_features = self._extract_custom_features(aggregated_obs)
        discrete_features = self._quantize_custom(continuous_features)
        return {
            "continuous": continuous_features,
            "discrete": discrete_features
        }
```

**Integration:**
```python
# In hot_restart.py
from dktm.custom_sosa import CustomSOSA
adapter = SOSAAdapter(sosa_algorithm=CustomSOSA())
```

---

### 2. Additional Platform Support

**Location**: `dktm/platform_*.py`

**How to Extend:**
```python
# Create platform_linux.py
class LinuxPlatformOperations(PlatformOperations):
    def commit_transition(self, auto_reboot=False, dry_run=False):
        # Use kexec for Linux
        if not dry_run:
            subprocess.run(["kexec", "-e"])

    def reboot(self, delay_seconds=5, dry_run=False):
        if not dry_run:
            subprocess.run(["shutdown", "-r", f"+{delay_seconds//60}"])
```

**Registration:**
```python
# In platform_ops.py
def get_platform_operations(config, dry_run=False):
    platform = sys.platform
    if platform == "linux":
        from .platform_linux import LinuxPlatformOperations
        return LinuxPlatformOperations(config, dry_run)
```

---

### 3. Custom Transition Phases

**Location**: `dktm/command_dict.py`

**How to Extend:**
```python
# Add custom phase
COMMAND_GROUPS = {
    "quiesce": [...],
    "flush": [...],
    "commit": [...],
    "custom_phase": [
        "custom_command_1",
        "custom_command_2"
    ]
}

COMMANDS = {
    "custom_command_1": {
        "description": "Custom operation",
        "priority": 100
    }
}
```

**Executor Integration:**
```python
# In executor.py
def execute_custom_phase(self, commands):
    for cmd in commands:
        self._execute_command(cmd)
```

---

### 4. Alternative State Detectors

**Location**: `dktm/retina_probe.py`

**How to Extend:**
```python
# Create ml_probe.py
def ml_probe(state_matrix):
    """Machine learning-based state detection."""
    import sklearn
    # Your ML model for system state classification
    model = load_pretrained_model()
    features = extract_features(state_matrix)
    prediction = model.predict(features)

    return {
        "E_mean": prediction["pressure"],
        "confidence": prediction["confidence"],
        "ml_features": features
    }
```

**Usage:**
```python
# In hot_restart.py
from dktm.ml_probe import ml_probe
result = ml_probe(state)
```

---

### 5. Custom Recovery Scripts

**Location**: WinPE image (injected by `build_pe.py`)

**How to Extend:**
```python
# In build_pe.py
def _create_custom_recovery_script(self):
    script = """
@echo off
REM Custom DKTM Recovery

echo [CUSTOM] Running extended diagnostics
wmic diskdrive get status
wmic memorychip get capacity

echo [CUSTOM] Custom cleanup operations
del /f /q C:\\Windows\\Temp\\*.*

REM Standard DKTM cleanup
bcdedit /deletevalue {bootmgr} bootsequence
wpeutil reboot
"""
    return script
```

---

### 6. Hooks and Callbacks

**Proposed Extension** (not yet implemented):

```python
# In dktm/dktm.py
class DKTM:
    def __init__(self, config, hooks=None):
        self.hooks = hooks or {}

    def _run_hook(self, hook_name, context):
        if hook_name in self.hooks:
            self.hooks[hook_name](context)

    def hot_restart(self):
        self._run_hook("before_probe", {})
        state = self.probe_system_state()
        self._run_hook("after_probe", {"state": state})

        # ... rest of logic
```

**Usage:**
```python
def custom_before_probe(context):
    print("About to probe system...")

dktm = DKTM(config, hooks={
    "before_probe": custom_before_probe
})
```

---

## Design Decisions

### Why BCD /bootsequence instead of /displayorder?

**Decision**: Use `/bootsequence` for one-time PE boot.

**Rationale:**
- `/bootsequence` is automatically cleared after one boot
- Safer: No risk of permanently changing boot order
- Official Windows API behavior
- Auto-cleanup even if PE script fails

**Alternative Rejected**: Using `/displayorder` would require manual cleanup and risk boot loop.

---

### Why Separate build_pe.py instead of inline?

**Decision**: Create standalone `tools/build_pe.py` script.

**Rationale:**
- Reusability: Can be run independently
- Testability: Easier to test PE build in isolation
- Modularity: Keeps installer.py focused on orchestration
- Debugging: Users can re-run PE build without full reinstall

---

### Why YAML for configuration?

**Decision**: Use YAML for `dktm_config.yaml`.

**Rationale:**
- Human-readable (users can inspect/edit)
- Comments support (for guidance)
- Nested structures (complex config)
- Standard in DevOps tools

**Alternative Rejected**: JSON (no comments), INI (limited nesting)

---

### Why Platform Abstraction Layer?

**Decision**: Create `platform_ops.py` abstraction.

**Rationale:**
- Development on non-Windows: POSIX stub enables macOS/Linux dev
- Testing: Can mock platform operations
- Future-proofing: Easier to add Linux kexec support later
- Clean separation: Business logic independent of platform

---

## Glossary

| Term | Definition |
|------|------------|
| **DKTM** | Dynamic Kernel Transition Mechanism |
| **SOSA** | Self-Organizing State Aggregator (custom algorithm) |
| **Binary-Twin** | SOSA output: continuous + discrete state representation |
| **Retina Probe** | Gradient-based state detector ("Jerry checking") |
| **E_mean** | Average system pressure (0-1 scale) |
| **explore_factor** | SOSA exploration coefficient (0-1 scale) |
| **WinPE** | Windows Preinstallation Environment |
| **BCD** | Boot Configuration Data (Windows boot settings) |
| **bootsequence** | BCD one-time boot order |
| **ADK** | Windows Assessment and Deployment Kit |
| **copype** | ADK tool for creating base WinPE |
| **DISM** | Deployment Image Servicing and Management |
| **bcdedit** | Command-line tool for BCD manipulation |
| **Marker File** | JSON file indicating DKTM transition in progress |
| **Dry-Run** | Simulation mode (logs actions without executing) |
| **Jerry探头** | "Jerry checking" - metaphor for safe state detection |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-26 | Initial schema documentation |

---

**End of Schema Documentation**
