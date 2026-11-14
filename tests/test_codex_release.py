"""Static release checks only: parse the script without executing it or Docker."""

from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "release.ps1"


class ReleaseScriptTests(unittest.TestCase):
    def test_build_context_names_match_dockerfile_exactly(self):
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        script = SCRIPT.read_text(encoding="utf-8")
        expected = re.findall(r"^\s*COPY\s+--from=([A-Za-z_][A-Za-z_0-9-]*)\b", dockerfile, re.M)
        actual = re.findall(r"""['"]--build-context['"]\s*,\s*["']([A-Za-z_][A-Za-z_0-9-]*)=""", script)
        self.assertEqual(len(expected), 2)
        self.assertEqual(len(actual), len(set(actual)), "Duplicate release build contexts")
        self.assertCountEqual(actual, expected)
        self.assertIn("'build'", script)
        self.assertIn("& docker @buildArguments", script)
        self.assertIn("& docker save --output $tarPath $ImageTag", script)
        self.assertIn("Get-FileHash -LiteralPath $tarPath -Algorithm MD5", script)
        self.assertIn('$md5Path = "$tarPath.md5"', script)
        self.assertIn("'{{.Size}}'", script)

    def test_validates_both_model_directories_before_build(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Resolve-ModelDirectory -Path $PlannerDir", script)
        self.assertIn("Resolve-ModelDirectory -Path $AsrDir", script)
        self.assertIn("Join-Path $resolved 'config.json'", script)
        self.assertIn("-File -Filter '*.safetensors'", script)
        self.assertNotIn("-Recurse", script)
        self.assertLess(script.index("$plannerPath = Resolve-ModelDirectory"), script.index("& docker @buildArguments"))
        self.assertLess(script.index("$asrPath = Resolve-ModelDirectory"), script.index("& docker @buildArguments"))
        self.assertIn("Refusing to overwrite", script)
        self.assertIn("if ($LASTEXITCODE -ne 0)", script)

    @unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell parser unavailable")
    def test_powershell_51_syntax_and_build_context_ast(self):
        # ParseFile does not execute release.ps1 or invoke Docker.
        path = str(SCRIPT).replace("'", "''")
        command = (
            "$tokens=$null; $parseErrors=$null; "
            f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
            "[ref]$tokens,[ref]$parseErrors); "
            "if ($parseErrors.Count -gt 0) { $parseErrors | ForEach-Object { "
            "Write-Output $_.Message }; exit 1 }; "
            "$operators=@($tokens | Where-Object { $_.Text -eq '&&' -or $_.Text -eq '||' }); "
            "if ($operators.Count -gt 0) { Write-Output 'Unsupported PS 5.1 operator'; exit 1 }; "
            "$contexts=@($ast.FindAll({ param($n) "
            "($n -is [System.Management.Automation.Language.StringConstantExpressionAst] "
            "-or $n -is [System.Management.Automation.Language.ExpandableStringExpressionAst]) "
            "-and $n.Value -match '^[a-z_]+_weights=' }, $true)); "
            "if ($contexts.Count -ne 2) { Write-Output 'Expected two build context values'; exit 1 }; "
            "Write-Output 'PowerShell parse OK'"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PowerShell parse OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
