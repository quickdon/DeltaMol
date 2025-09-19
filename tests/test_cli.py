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
