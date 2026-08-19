#!/usr/bin/env python3
"""Validate the skill package without importing optional AI, vision or browser stacks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FILES = (
    "SKILL.md",
    "templates/core_modules/langgraph_engine.py",
    "templates/core_modules/vision_engine.py",
    "templates/core_modules/automation_engine.py",
    "templates/core_modules/rag_engine_enhanced.py",
    "templates/core_modules/core_integration.py",
    "templates/core_modules/test_new_modules.py",
    "templates/report_template.md",
    "scripts/visualize_report.py",
)


def validate(root: Path) -> dict:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    skill_file = root / "SKILL.md"
    content = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""
    front_matter_ok = content.startswith("---\n") and "name: ai-capability-gap-filler" in content
    return {
        "root": str(root),
        "valid": not missing and front_matter_ok,
        "front_matter_ok": front_matter_ok,
        "missing_files": missing,
        "checked_files": list(REQUIRED_FILES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate skill structure without optional runtime dependencies.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = validate(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("skill integrity:", "PASS" if result["valid"] else "FAIL")
        if result["missing_files"]:
            print("missing:", ", ".join(result["missing_files"]))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
