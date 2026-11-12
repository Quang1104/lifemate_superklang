"""Keep direct filesystem access away from environment data."""

from pathlib import Path
import os
import stat


def safe_path(path: str | Path) -> Path:
    path = Path(os.path.abspath(path))
    for part in path.parts:
        folded = part.casefold().rstrip(" .")
        if folded == "state.json" or folded.endswith("_environment"):
            raise ValueError("This path is reserved for the official ToolEnv")
    # Reject link indirection before following any component of a supplied path.
    for component in (*reversed(path.parents), path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            break
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_reparse_tag", 0) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)):
            raise ValueError("Symbolic links and junctions are not supported here")
    return path
