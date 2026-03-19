#!/usr/bin/env python3
"""
DKTM Code Quality Checker
==========================

Checks for common bugs and issues in DKTM codebase.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

def check_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Check a Python file for common issues.

    Returns list of (line_number, severity, message).
    """
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return [(e.lineno or 0, "ERROR", f"Syntax error: {e.msg}")]

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['numpy', 'yaml', 'pyyaml']:
                        issues.append((node.lineno, "WARNING",
                            f"Import '{alias.name}' may not be available - check dependencies"))

            # Check encoding in file operations
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['write_text', 'open']:
                        # Check for 'ansi' encoding
                        for keyword in node.keywords:
                            if keyword.arg == 'encoding':
                                if isinstance(keyword.value, ast.Constant):
                                    if keyword.value.value == 'ansi':
                                        issues.append((node.lineno, "ERROR",
                                            "Invalid encoding 'ansi' - use 'cp1252' or 'mbcs' instead"))

        # Check for hardcoded paths
        if 'C:\\' in content or 'C:/' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'C:\\' in line and 'r"C:\\' not in line and '"C:\\\\' not in line:
                    issues.append((i, "WARNING", "Hardcoded Windows path - ensure proper escaping"))

    except Exception as e:
        issues.append((0, "ERROR", f"Failed to check file: {e}"))

    return issues

def main():
    """Main entry point."""
    project_root = Path(__file__).parent

    files_to_check = [
        project_root / "hot_restart.py",
        project_root / "install.py",
        project_root / "tools" / "build_pe.py",
        project_root / "tools" / "setup_bcd.py",
    ]

    print("🔍 DKTM Code Quality Check\n")

    total_issues = 0
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"⚠️  {file_path.name}: File not found")
            continue

        issues = check_file(file_path)

        if issues:
            print(f"📄 {file_path.name}:")
            for line, severity, message in issues:
                emoji = "❌" if severity == "ERROR" else "⚠️ "
                print(f"  {emoji} Line {line}: {message}")
                total_issues += 1
            print()
        else:
            print(f"✅ {file_path.name}: No issues found")

    print(f"\nTotal issues found: {total_issues}")
    return 0 if total_issues == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
