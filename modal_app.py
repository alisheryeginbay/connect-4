"""Run data generation and training on Modal.

    uv run modal run modal_app.py::generate --name train_5m --shards 50
    uv run modal run modal_app.py::generate --name probe_100k --shards 1 --first-seed 1000000 --exclude train_5m
    uv run modal run modal_app.py::train --data train_5m --run base
    uv run modal run modal_app.py::train --data train_5m --run end --end-token
    uv run modal volume get connect4 runs/base runs/      # download the checkpoint

Everything is stored on the "connect4" Volume: data/<name>/shard_<seed>.npy and runs/<run>/.
Shard i of a dataset uses seed first_seed + i, so keep the seed ranges of different
datasets apart (training: 0, 1, 2, ...; probe: 1000000, ...).
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
def generate_shard(name: str, seed: int, n: int) -> str:
    from pathlib import Path

    import numpy as np

    from connect4.generate import generate, shard_name

    path = Path(VOL) / "data" / name / shard_name(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, generate(n, seed))
    volume.commit()
    return str(path)


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 60, memory=8192)
def drop_overlap(name: str, exclude: str) -> None:
    """Remove from every shard of `name` the games that also appear anywhere in `exclude`."""
    from pathlib import Path

    import numpy as np

    from connect4.generate import load_games, remove_overlap

    volume.reload()
    excluded = load_games(Path(VOL) / "data" / exclude)
    for path in sorted((Path(VOL) / "data" / name).glob("shard_*.npy")):
        games = np.load(path)
        kept = remove_overlap(games, excluded)
        np.save(path, kept)
        print(f"{path.name}: dropped {len(games) - len(kept)} games that also appear in {exclude}")
    volume.commit()


@app.local_entrypoint()
def generate(name: str, shards: int, shard_size: int = 100_000, first_seed: int = 0, exclude: str = "") -> None:
    seeds = range(first_seed, first_seed + shards)
    # Each shard runs in its own container, so 50 shards take about as long as one.
    for path in generate_shard.map([name] * shards, seeds, [shard_size] * shards):
        print("saved", path)
    if exclude:
        drop_overlap.remote(name, exclude)


@app.function(image=image, volumes={VOL: volume}, gpu="L40S", timeout=6 * 60 * 60)
def train(data: str, run: str, end_token: bool = False, epochs: int = 2) -> None:
    from connect4.train import TrainConfig
    from connect4.train import train as train_model

    cfg = TrainConfig(data=f"{VOL}/data/{data}", out=f"{VOL}/runs/{run}", end_token=end_token, epochs=epochs)
    train_model(cfg)
    volume.commit()
