import logging
import glob
from pathlib import Path

log = logging.getLogger(__name__)

def find_file_paths(file_pattern: str) -> list[Path]:
    """Finds and sorts files matching a glob pattern."""
    log.info(f"Searching for files with pattern: '{file_pattern}'")
    file_paths = sorted(Path(p) for p in glob.glob(file_pattern))
    log.info(f"Found {len(file_paths)} file(s).")
    return file_paths
