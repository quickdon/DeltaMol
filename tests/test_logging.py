from deltamol.utils.logging import configure_logging


def test_configure_logging_creates_timestamped_log(tmp_path):
    configure_logging(tmp_path)

    archived = list(tmp_path.glob("training-*.log"))
    assert len(archived) == 1
    assert archived[0].read_text() == ""
    assert not (tmp_path / "training.log").exists()


def test_configure_logging_resume_uses_latest_timestamped_log(tmp_path):
    first_log = tmp_path / "training-20240101-000000.log"
    first_log.write_text("keep me")
    later_log = tmp_path / "training-20240102-000000.log"
    later_log.write_text("append here")

    configure_logging(tmp_path, resume=True)

    archived = sorted(tmp_path.glob("training-*.log"))
    assert len(archived) == 2
    assert archived[-1].read_text().startswith("append here")
