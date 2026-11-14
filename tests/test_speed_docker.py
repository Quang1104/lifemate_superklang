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
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "env"
        and call.func.attr == "setdefault"
    }
    assert script_defaults["LIFEMATE_PLANNER_MODE"] == notebook_defaults["LIFEMATE_PLANNER_MODE"] == "native"
    model_var = "LIFEMATE_PLANNER_MODEL_DIR"
    model_dir = script_defaults[model_var]
    assert model_dir == notebook_defaults[model_var] == "/models/planner"
    assert '--model-dir "${' + model_var + ':-' + model_dir + '}"' in script

    runs = [
        call for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr == "run"
    ]
    assert len(runs) == 1
    assert ast.literal_eval(runs[0].args[0]) == [
        "bash", "/code/predict.sh", "/result/jupyter_submission.jsonl"
    ]
    assert any(keyword.arg == "env" and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "env" for keyword in runs[0].keywords)
    dockerfile = (docker_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG PLANNER_MODE=native" in dockerfile
    assert f"COPY --from=planner_weights . {model_dir}/" in dockerfile
