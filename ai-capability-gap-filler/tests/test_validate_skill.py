import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_skill.py"
SPEC = importlib.util.spec_from_file_location("validate_skill", MODULE_PATH)
validate_skill = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validate_skill)


class SkillIntegrityTests(unittest.TestCase):
    def test_current_skill_package_is_complete(self) -> None:
        result = validate_skill.validate(ROOT)
        self.assertTrue(result["valid"])
        self.assertTrue(result["front_matter_ok"])
        self.assertEqual(result["missing_files"], [])

    def test_missing_root_is_reported_without_optional_dependencies(self) -> None:
        result = validate_skill.validate(ROOT / "does-not-exist")
        self.assertFalse(result["valid"])
        self.assertIn("SKILL.md", result["missing_files"])


if __name__ == "__main__":
    unittest.main()
