from deltamol.utils.logging import configure_logging


def test_configure_logging_archives_existing(tmp_path):
    log_path = tmp_path / "training.log"
    log_path.write_text("previous run")

    configure_logging(tmp_path)

    archived = list(tmp_path.glob("training-*.log"))
    assert archived
    assert archived[0].read_text() == "previous run"
    assert log_path.exists()
    assert log_path.read_text() == ""


def test_configure_logging_resume_keeps_existing_log(tmp_path):
    log_path = tmp_path / "training.log"
    log_path.write_text("keep me")

    configure_logging(tmp_path, resume=True)

    assert not list(tmp_path.glob("training-*.log"))
    assert log_path.read_text().startswith("keep me")
