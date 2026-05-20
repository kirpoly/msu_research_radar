from msu_research_radar.cli import main


def test_cli_main_help_runs() -> None:
    assert main([]) == 0
