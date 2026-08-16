# Amendment: fix recovery region semantics

Page-level Qwen text without coordinates must remain page-level recovery data,
not synthetic full-page spatial regions. It must not trigger TrOCR or false
disagreements. Diagram entries with legitimate normalized boxes become drawing
regions and original-source assets. Recovery JSON must expose page-level text
separately from spatial regions, and benchmark metadata must report zero
spatial text regions and zero TrOCR attempts for the supplied response.
