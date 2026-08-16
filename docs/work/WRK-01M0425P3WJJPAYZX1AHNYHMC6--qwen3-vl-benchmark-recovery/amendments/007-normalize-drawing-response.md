# Amendment: normalize drawing response variants

The drawing-localization adapter must inspect valid top-level `drawings` JSON
and normalize only established drawing records into the internal normalized
`bbox_2d` plus optional description representation. The existing normalized
`bbox` fixture form is accepted as a compatibility alias; arbitrary drawing
entry shapes remain failures with structural diagnostics. Page transform,
prompting, coordinate mapping, padding, crops, and projections are unchanged.
