from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
from pathlib import Path
import time
import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from MulaTOVA_MNN import TopologyOptimizer

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "struct"

app_state = {
    "boundary_conditions": [],
    "loading_conditions": [],
    "result_message": None,
    "images": {},
    "selected_result_base": None,
}


def load_config():
    pipe = ConfigPipeline(
        [
            YamlConfig(
                str(BASE_DIR / "config" / "struct.yaml"),
                config_name="default",
                config_folder=str(BASE_DIR / "config"),
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder=str(BASE_DIR / "config")),
        ]
    )
    return pipe.read_conf()


def build_image_paths(base: str):
    stamp = str(time.time())
    return {
        "convergence": f"/results/{base}_convergence.png?v={stamp}",
        "topology": f"/results/{base}_topology.jpg?v={stamp}",
        "true_disp": f"/results/{base}_true_displacement.jpg?v={stamp}",
        "normalized_disp": f"/results/{base}_normalized_displacement.jpg?v={stamp}",
        "target_disp": f"/results/{base}_target_displacement.jpg?v={stamp}",
    }


def discover_result_bases():
    if not RESULTS_DIR.exists():
        return []

    suffixes = [
        "_topology.jpg",
        "_convergence.png",
        "_true_displacement.jpg",
        "_normalized_displacement.jpg",
        "_target_displacement.jpg",
    ]

    found = set()
    for path in RESULTS_DIR.iterdir():
        if not path.is_file():
            continue
        for suffix in suffixes:
            if path.name.endswith(suffix):
                found.add(path.name[:-len(suffix)])

    return sorted(found, reverse=True)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Topology Optimization Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }
        h1, h2, h3 {
            color: #222;
        }
        .panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        input, select {
            padding: 8px;
            margin: 5px;
            width: 130px;
        }
        button {
            padding: 10px 15px;
            margin-top: 10px;
            margin-right: 10px;
            background: #2d6cdf;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        button:hover {
            background: #1f56b5;
        }
        .danger {
            background: #c0392b;
        }
        .danger:hover {
            background: #962d22;
        }
        .secondary {
            background: #777;
        }
        .secondary:hover {
            background: #555;
        }
        ul {
            margin-top: 12px;
        }
        li {
            margin-bottom: 6px;
        }
        .images-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .img-box {
            background: #fafafa;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        img {
            max-width: 430px;
            border-radius: 8px;
            display: block;
        }
        .result-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .result-card {
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 12px;
            min-width: 280px;
        }
        .small {
            color: #666;
            font-size: 14px;
        }
        .inline-form {
            display: inline-block;
        }
    </style>
</head>
<body>

<h1>Topology Optimization Interface</h1>

<div class="panel">
    <h2>Boundary Conditions</h2>
    <form method="post" action="/add_boundary">
        x: <input type="number" name="x" required>
        y: <input type="number" name="y" required>
        <br>
        <button type="submit">Add Boundary</button>
    </form>

    <form method="post" action="/clear_boundaries" class="inline-form">
        <button type="submit" class="danger">Clear Boundary Conditions</button>
    </form>

    <ul>
    {% if boundary_conditions %}
        {% for bc in boundary_conditions %}
            <li>(x={{ bc.x }}, y={{ bc.y }})</li>
        {% endfor %}
    {% else %}
        <li>No boundary conditions added.</li>
    {% endif %}
    </ul>
</div>

<div class="panel">
    <h2>Loading Conditions</h2>
    <form method="post" action="/add_load">
        fx: <input type="number" step="any" name="fx" required>
        fy: <input type="number" step="any" name="fy" required>
        x: <input type="number" name="x" required>
        y: <input type="number" name="y" required>
        <br>
        <button type="submit">Add Load</button>
    </form>

    <form method="post" action="/clear_loads" class="inline-form">
        <button type="submit" class="danger">Clear Loading Conditions</button>
    </form>

    <ul>
    {% if loading_conditions %}
        {% for l in loading_conditions %}
            <li>(fx={{ l.fx }}, fy={{ l.fy }}, x={{ l.x }}, y={{ l.y }})</li>
        {% endfor %}
    {% else %}
        <li>No loading conditions added.</li>
    {% endif %}
    </ul>
</div>

<div class="panel">
    <h2>Run Settings</h2>

    <form method="post" action="/run">
        <label>Target Shape:</label>
        <select name="target_type">
            <option value="x">X Shape</option>
            <option value="square">Square</option>
            <option value="circle">Circle</option>
        </select>

        <label>Square Start:</label>
        <input type="number" name="target_square_start" value="5">

        <label>Square End:</label>
        <input type="number" name="target_square_end" value="15">

        <br>
        <button type="submit">Run Topology Optimization</button>
    </form>

    <form method="post" action="/clear_all" class="inline-form">
        <button type="submit" class="danger">Clear All Conditions</button>
    </form>
</div>

{% if result_message %}
<div class="panel">
    <h2>Status</h2>
    <p>{{ result_message }}</p>
</div>
{% endif %}

<div class="panel">
    <h2>Past Topology Results</h2>
    <p class="small">Detected automatically from <code>results/struct</code>.</p>

    {% if result_bases %}
        <div class="result-list">
            {% for base in result_bases %}
                <div class="result-card">
                    <strong>{{ base }}</strong>
                    {% if selected_result_base == base %}
                        <p class="small">Currently selected</p>
                    {% endif %}
                    <form method="post" action="/load_result/{{ base|urlencode }}">
                        <button type="submit" class="secondary">View This Result</button>
                    </form>
                </div>
            {% endfor %}
        </div>
    {% else %}
        <p>No saved result sets found.</p>
    {% endif %}
</div>

{% if images %}
<div class="panel">
    <h2>Displayed Results{% if selected_result_base %}: {{ selected_result_base }}{% endif %}</h2>

    <div class="images-row">
        {% if images.convergence %}
        <div class="img-box">
            <h3>Convergence</h3>
            <img src="{{ images.convergence }}">
        </div>
        {% endif %}

        {% if images.topology %}
        <div class="img-box">
            <h3>Topology</h3>
            <img src="{{ images.topology }}">
        </div>
        {% endif %}

        {% if images.true_disp %}
        <div class="img-box">
            <h3>True Displacement</h3>
            <img src="{{ images.true_disp }}">
        </div>
        {% endif %}

        {% if images.normalized_disp %}
        <div class="img-box">
            <h3>Normalized Displacement</h3>
            <img src="{{ images.normalized_disp }}">
        </div>
        {% endif %}

        {% if images.target_disp %}
        <div class="img-box">
            <h3>Target Displacement</h3>
            <img src="{{ images.target_disp }}">
        </div>
        {% endif %}
    </div>
</div>
{% endif %}

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        HTML,
        boundary_conditions=app_state["boundary_conditions"],
        loading_conditions=app_state["loading_conditions"],
        result_message=app_state["result_message"],
        images=app_state["images"],
        result_bases=discover_result_bases(),
        selected_result_base=app_state["selected_result_base"],
    )


@app.route("/add_boundary", methods=["POST"])
def add_boundary():
    app_state["boundary_conditions"].append(
        {
            "x": int(request.form["x"]),
            "y": int(request.form["y"]),
        }
    )
    app_state["result_message"] = "Boundary condition added."
    return redirect(url_for("home"))


@app.route("/add_load", methods=["POST"])
def add_load():
    app_state["loading_conditions"].append(
        {
            "fx": float(request.form["fx"]),
            "fy": float(request.form["fy"]),
            "x": int(request.form["x"]),
            "y": int(request.form["y"]),
        }
    )
    app_state["result_message"] = "Loading condition added."
    return redirect(url_for("home"))


@app.route("/clear_boundaries", methods=["POST"])
def clear_boundaries():
    app_state["boundary_conditions"] = []
    app_state["result_message"] = "Boundary conditions cleared."
    return redirect(url_for("home"))


@app.route("/clear_loads", methods=["POST"])
def clear_loads():
    app_state["loading_conditions"] = []
    app_state["result_message"] = "Loading conditions cleared."
    return redirect(url_for("home"))


@app.route("/clear_all", methods=["POST"])
def clear_all():
    app_state["boundary_conditions"] = []
    app_state["loading_conditions"] = []
    app_state["result_message"] = "All conditions cleared."
    return redirect(url_for("home"))


@app.route("/load_result/<path:base>", methods=["POST"])
def load_result(base):
    app_state["images"] = build_image_paths(base)
    app_state["selected_result_base"] = base
    app_state["result_message"] = f"Loaded saved result set: {base}"
    return redirect(url_for("home"))


@app.route("/run", methods=["POST"])
def run():
    try:
        # FIX: Register PytorchMinMaxScaler under __main__ so that torch.load()
        # can find it when deserializing any previously saved .nt network files,
        # regardless of which module is currently __main__.
        import utils as _utils
        sys.modules['__main__'].PytorchMinMaxScaler = _utils.PytorchMinMaxScaler

        config = load_config()

        # Match your original script behavior
        config.example = 7
        config.nn_type = "SIMP"
        config.results_dir = str(RESULTS_DIR)
        config.useSavedNet = False
        # New target parameters from app
        config.target_type = request.form.get("target_type", "x")
        config.target_square_start = int(request.form.get("target_square_start", 5))
        config.target_square_end = int(request.form.get("target_square_end", 15))

        start = time.perf_counter()

        top_opt = TopologyOptimizer(config)
        top_opt.optimizeDesign(config)
        top_opt.plotConvergence()

        elapsed = time.perf_counter() - start
        base = top_opt.exper_name

        print("APP BASE_DIR:", BASE_DIR)
        print("APP RESULTS_DIR:", RESULTS_DIR)
        print("CONFIG RESULTS_DIR:", config.results_dir)
        print("EXPER NAME:", base)
        print("TARGET TYPE:", config.target_type)

        app_state["images"] = build_image_paths(base)
        app_state["selected_result_base"] = base
        app_state["result_message"] = (
            f"Optimization finished in {elapsed:.2f} seconds. "
            f"Displaying result set: {base}. "
            f"Target shape: {config.target_type}"
        )

    except Exception as e:
        app_state["result_message"] = f"Run failed: {e}"

    return redirect(url_for("home"))


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


if __name__ == "__main__":
    app.run(debug=True)