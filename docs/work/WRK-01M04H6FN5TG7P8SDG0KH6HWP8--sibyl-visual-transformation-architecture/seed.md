# Sibyl architectural foundation: visual transformation decisions

Create one Work Item for this architectural phase and exactly six ADRs using
the installed `brfx adr` command. Do not implement product behavior, amend
existing ADRs, create additional ADRs, or run model inference.

ADR 1: Sibyl is a visual transformation system. Sibyl transforms visual
documents into structured representations; it is not architecturally an OCR
system. `sibyl run IMAGE` is canonical, with `transform.json` as the structured
representation, `transform.md` as a projection, and source-derived visual
artifacts retained where appropriate.

ADR 2: Text transformation and figure extraction are separate responsibilities
over the same source image. Page transformation owns page text; figure
localization owns figure geometry and source crops. Neither becomes the other’s
canonical mechanism.

ADR 3: Model implementations are subordinate to transformation contracts.
Sibyl owns responsibilities and contracts; Qwen or any other model is
replaceable implementation machinery beneath those contracts.

ADR 4: Visual artifacts preserve source pixels. When visual content can be
preserved directly, prefer source-resolution crops identified by localized
geometry over model-generated descriptions.

ADR 5: Source coordinates are authoritative for derived visual artifacts.
Model-space geometry is intermediate; deterministic conversion proceeds through
normalized and prepared-image coordinates to source-image coordinates and source
pixels.

ADR 6: Transformation preserves interpretation and uncertainty. Source-derived
content and model interpretation remain distinguishable; semantic plausibility
must not silently replace uncertain observed content (for example, `Sap` must
not silently become `Splice`).

The six decisions form one foundation: visual transformation → separate
responsibilities → replaceable model implementations → preserved visual
evidence → source-authoritative geometry → explicit interpretation uncertainty.

Use repository numbering, formatting, and BRFX Work Item conventions. The Work
Item must reference the six ADRs after creation. Validate with `just check`,
`sextant check`, `sextant audit`, `git diff --check`, and any ADR validation
provided by the repository. Do not change production implementation files.
