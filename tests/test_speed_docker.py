import ast
import json
import re
from pathlib import Path


def test_docker_and_notebook_share_native_planner_defaults():
    docker_dir = Path(__file__).resolve().parents[1] / "docker"
    script = (docker_dir / "predict.sh").read_text(encoding="utf-8")
    script_defaults = dict(re.findall(r"[$][{](LIFEMATE_[A-Z_]+):-([^}]+)[}]", script))
    notebook = json.loads((docker_dir / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    notebook_defaults = {
        ast.literal_eval(call.args[0]): ast.literal_eval(call.args[1])
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func.value) == "os.environ"
        and call.func.attr == "setdefault"
    }
    assert script_defaults["LIFEMATE_PLANNER_MODE"] == notebook_defaults["LIFEMATE_PLANNER_MODE"] == "native"
    model_var = "LIFEMATE_PLANNER_MODEL_DIR"
    model_dir = script_defaults[model_var]
    assert model_dir == notebook_defaults[model_var] == "/models/planner"
    assert '--model-dir "${' + model_var + ':-' + model_dir + '}"' in script

    # The notebook runs the same in-process path as the CLI, with planner mode from the env default.
    assert "run_private_test(models" in code and "planner_mode=os.environ['LIFEMATE_PLANNER_MODE']" in code
    dockerfile = (docker_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG PLANNER_MODE=native" in dockerfile
    assert f"COPY --from=planner_weights . {model_dir}/" in dockerfile
