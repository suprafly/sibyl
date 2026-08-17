# Amendment: BOOX-assisted page recovery

Add a human-owned page recovery path that consumes the page image and verified
BOOX note, automatically enumerates line targets, uses same-region references
with structural target exclusion, and writes a deterministic `recovery.md` plus
machine-readable recovery evidence. Keep canonical `transform.md` image-only;
the earlier compatibility alias is not part of the accepted design.
