"""Static checks of the release/cleanup scripts: parsed, never executed, Docker never invoked."""

from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "docker" / "release.ps1"
CLEANUP = ROOT / "docker" / "cleanup.ps1"


class ReleaseScriptTests(unittest.TestCase):
    def test_build_contexts_match_dockerfile(self):
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        script = RELEASE.read_text(encoding="utf-8")
        expected = set(re.findall(r"--mount=type=bind,from=([A-Za-z_][\w-]*)", dockerfile))
        expected |= set(re.findall(r"^\s*COPY\s+--from=([A-Za-z_][\w-]*)", dockerfile, re.M))
        actual = set(re.findall(r"""['"]--build-context['"]\s*,\s*["']([A-Za-z_][\w-]*)=""", script))
        self.assertEqual(actual, expected)
        self.assertEqual(expected, {"adapter"})

    def test_release_checks_inputs_and_writes_archive_outside_the_repo(self):
        script = RELEASE.read_text(encoding="utf-8")
        for needle in ("adapter_config.json", "adapter_model.safetensors", "Refusing to overwrite",
                       "if ($LASTEXITCODE -ne 0)", "GZipStream", "Get-FileHash -LiteralPath $OutputArchive -Algorithm MD5",
                       "images_before", "$env:USERPROFILE 'lifemate_release"):
            self.assertIn(needle, script)
        self.assertLess(script.index("adapter_model.safetensors"), script.index("& docker @buildArguments"))

    def test_cleanup_only_removes_images_added_after_the_recorded_list(self):
        script = CLEANUP.read_text(encoding="utf-8")
        self.assertIn("$before -notcontains $id", script)
        self.assertIn("release-state.json", script)
        self.assertIn("-DeleteArchive", script)  # the archive is kept unless asked

    @unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell parser unavailable")
    def test_scripts_parse_under_powershell_51(self):
        for path in (RELEASE, CLEANUP):
            quoted = str(path).replace("'", "''")
            command = (
                "$tokens=$null; $errors=$null; "
                f"[void][System.Management.Automation.Language.Parser]::ParseFile('{quoted}',[ref]$tokens,[ref]$errors); "
                "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Output $_.Message }; exit 1 }; "
                "if (@($tokens | Where-Object { $_.Text -in '&&','||' }).Count) { Write-Output 'PS7-only operator'; exit 1 }; "
                "Write-Output 'parse OK'"
            )
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"{path.name}: {result.stdout}{result.stderr}")


if __name__ == "__main__":
    unittest.main()
