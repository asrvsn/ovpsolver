"""The form-compilation cache, and the one way an interrupted run damages it.

FFCx claims a module name by creating its ``.c`` file exclusively. That file is
a lock, not the cache: the compiled extension beside it is what a later run
loads, and a ``.c.cached`` marker written last says the two are a finished pair.
A failed compile renames its claim to ``.c.failed`` and frees the name.

Ctrl-C during compilation -- the slowest part of starting a run, so the likeliest
moment to interrupt one -- leaves a claim nothing will finish. Every later run
needing that module waits out the JIT timeout for the marker and raises
``TimeoutError``, and a mixture needs all of its hundred-odd modules. FFCx cannot
tell an abandoned claim from a slow one; a caller can, because it knows how long
is too long.
"""

from __future__ import annotations

import time
from pathlib import Path

from dolfinx.jit import get_options

#: How long a claim must have gone unfinished before it is taken as abandoned.
#:
#: Not the JIT timeout: a claim older than that is already fatal to anyone
#: waiting on it, but its owner may still be about to write the marker. So this
#: is well clear of any single form's compile time, at the cost that a run
#: restarted within the window pays one timeout before the next attempt succeeds.
ABANDONED_AFTER = 60.0


def release_abandoned_forms(older_than: float = ABANDONED_AFTER) -> list[Path]:
    """Free the cache entries an interrupted compile left claimed.

    Returns what was released, so a caller can report it rather than quietly
    delete from a directory it does not own.

    Safe because a claim carries nothing: the source is regenerated on the next
    attempt, the compiled extension is untouched, and an entry with its marker is
    never considered. What a claim might have is a live owner, which is what
    ``older_than`` is for.
    """

    cache = Path(get_options()["cache_dir"]).expanduser()
    if not cache.is_dir():
        return []

    cutoff = time.time() - float(older_than)
    released = []
    for claim in cache.glob("*.c"):
        if claim.with_suffix(".c.cached").exists():
            continue
        try:
            if claim.stat().st_mtime > cutoff:
                continue
            claim.unlink()
        except OSError:
            # Removed by somebody else between the glob and here, or not ours to
            # remove. Either way there is nothing to report and nothing to fix.
            continue
        released.append(claim)
    return released


__all__ = ["ABANDONED_AFTER", "release_abandoned_forms"]
