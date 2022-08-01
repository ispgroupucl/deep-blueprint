from functools import lru_cache
from subprocess import DEVNULL, CalledProcessError, check_output


@lru_cache()
def call_command(*command):
    return check_output(command, stderr=DEVNULL).decode().strip()


def git_info(info="commit", git_dir="."):
    valid_infos = ("commit", "branch")
    if info == "commit":
        return call_command("git", "-C", git_dir, "rev-parse", "HEAD")
    elif info == "branch":
        return call_command(
            "git",
            "-C",
            git_dir,
            "rev-parse",
            "--symbolic-full-name",
            "--abbrev-ref",
            "HEAD",
        )
    else:
        raise ValueError(f"info should be in {valid_infos}, not '{info}'")
