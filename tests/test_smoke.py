import gesturelink
from gesturelink.cli import main


def test_version():
    assert gesturelink.__version__ == "0.1.0"


def test_help_without_command(capsys):
    assert main([]) == 0
    assert "info" in capsys.readouterr().out
