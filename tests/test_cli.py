import pytest

pytest.importorskip("torch")

from deltamol.cli import build_parser


def test_cli_parses_train_command():
    parser = build_parser()
    args = parser.parse_args(["train-baseline", "dataset.npz"])
    assert callable(args.func)


def test_cli_parses_descriptor_command():
    parser = build_parser()
    args = parser.parse_args(
        ["cache-descriptors", "dataset.npz", "--descriptor", "soap"]
    )
    assert callable(args.func)
    assert args.descriptor == "soap"


def test_cli_parses_potential_model_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-potential",
            "dataset.npz",
            "--config",
            "experiment.yaml",
            "--model",
            "se3",
            "--hidden-dim",
            "96",
            "--transformer-layers",
            "3",
            "--se3-layers",
            "2",
            "--se3-distance-embedding",
            "48",
            "--num-heads",
            "6",
            "--ffn-dim",
            "256",
            "--dropout",
            "0.2",
            "--cutoff",
            "4.5",
            "--predict-forces",
            "--residual-mode",
        ]
    )
    assert callable(args.func)
    assert args.model_name == "se3"
    assert args.se3_layers == 2
    assert args.predict_forces is True


def test_cli_parses_gemnet_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-potential",
            "dataset.npz",
            "--config",
            "experiment.yaml",
            "--model",
            "gemnet",
            "--gemnet-num-blocks",
            "2",
            "--gemnet-num-radial",
            "8",
            "--gemnet-num-spherical",
            "6",
        ]
    )

    assert callable(args.func)
    assert args.model_name == "gemnet"
    assert args.gemnet_num_blocks == 2
    assert args.gemnet_num_radial == 8
    assert args.gemnet_num_spherical == 6
