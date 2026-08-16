from __future__ import annotations

import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_manager():
    loader = importlib.machinery.SourceFileLoader("container_manager", str(ROOT / "c"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


manager = load_manager()


class ManagerTests(unittest.TestCase):
    def test_configuration_is_kept_out_of_the_repository_root(self) -> None:
        self.assertEqual(manager.ENV_FILE, ROOT / ".docker" / ".env")
        self.assertEqual(manager.COMPOSE_FILE, ROOT / ".docker" / "compose.yml")
        self.assertEqual(manager.RUNTIME_DIR, ROOT / ".docker" / ".runtime")
        self.assertEqual(manager.PYTHON_PROJECT_DIR, ROOT / ".python_env")

    def test_aliases_point_to_existing_commands(self) -> None:
        for command in manager.ALIASES.values():
            self.assertIn(command, manager.COMMANDS)

    def test_native_environment_commands_exist(self) -> None:
        self.assertIn("native-sync", manager.COMMANDS)
        self.assertIn("native-check", manager.COMMANDS)

    def test_project_slug_is_compose_safe(self) -> None:
        self.assertRegex(manager.project_slug(), r"^[a-z0-9][a-z0-9-]*$")

    def test_replace_env_values_preserves_comments(self) -> None:
        source = "# comment\nTOKEN=old\nUNCHANGED=yes\n"
        result = manager.replace_env_values(source, {"TOKEN": "new"})
        self.assertEqual(result, "# comment\nTOKEN=new\nUNCHANGED=yes\n")

    def test_parse_env_keeps_spaces_in_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("APP_COMMAND=python examples/worker.py\n", encoding="utf-8")
            self.assertEqual(manager.parse_env(path)["APP_COMMAND"], "python examples/worker.py")

    def test_public_key_validation_rejects_private_or_empty_data(self) -> None:
        with self.assertRaises(manager.CommandError):
            manager.validate_public_keys("")
        with self.assertRaises(manager.CommandError):
            manager.validate_public_keys("-----BEGIN OPENSSH PRIVATE KEY-----")

    def test_public_key_validation_accepts_ed25519(self) -> None:
        manager.validate_public_keys("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest user@example\n")


if __name__ == "__main__":
    unittest.main()
