"""Run data generation and training on Modal.

    uv run modal run modal_app.py::generate --n 5000000 --name games_5m
    uv run modal run modal_app.py::train --data games_5m --run base
    uv run modal run modal_app.py::train --data games_5m --run end --end-token
    uv run modal volume get connect4 runs/base runs/      # download the checkpoint

Everything is stored on the "connect4" Volume: data/<name>.npy and runs/<run>/.
"""

import modal

app = modal.App("connect4")
volume = modal.Volume.from_name("connect4", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "numpy==2.5.3")
    .add_local_python_source("connect4")
)


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 60)
def generate(n: int, name: str, seed: int = 0) -> None:
    from pathlib import Path

    import numpy as np

    from connect4.generate import generate as generate_games

    path = Path(VOL) / "data" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, generate_games(n, seed))
    volume.commit()
    print(f"saved {n} games to {path}")


@app.function(image=image, volumes={VOL: volume}, gpu="L40S", timeout=6 * 60 * 60)
def train(data: str, run: str, end_token: bool = False, epochs: int = 2) -> None:
    from connect4.train import TrainConfig
    from connect4.train import train as train_model

    cfg = TrainConfig(data=f"{VOL}/data/{data}.npy", out=f"{VOL}/runs/{run}", end_token=end_token, epochs=epochs)
    train_model(cfg)
    volume.commit()
