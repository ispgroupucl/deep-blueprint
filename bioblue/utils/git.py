from functools import lru_cache
from subprocess import DEVNULL, check_output


@lru_cache()
def call_command(*command):
    return check_output(command, stderr=DEVNULL).decode().strip()


def git_info(info="commit"):
    valid_infos = ("commit", "branch")
    if info == "commit":
        return call_command("git", "rev-parse", "HEAD")
    elif info == "branch":
        return call_command("git", "rev-parse", "--symbolic-full-name", "--abbrev-ref", "HEAD")
    else:
        raise ValueError(f"info should be in {valid_infos}, not '{info}'")
