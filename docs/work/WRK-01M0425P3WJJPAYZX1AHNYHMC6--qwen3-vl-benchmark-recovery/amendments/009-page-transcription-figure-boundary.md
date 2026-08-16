# Amendment: separate page transcription from figure content

The page-recovery prompt must assign ordinary handwritten notes to `page_text`
and visual figure content to the independent drawing-localization pipeline. It
must exclude diagram marks, arrows, strokes, connectors, isolated figure
symbols, and figure-attached annotations from page text. It must require
faithful transcription of visible letterforms, including unfamiliar technical
vocabulary, while forbidding semantic correction and using `[unclear]` only for
genuinely unreadable text.

Only the page-recovery prompt and its mocked regression coverage are in scope.
Drawing localization, structured response handling, coordinate mapping,
padding, crops, Markdown projection, TrOCR behavior, image preparation, and
artifact layout remain unchanged. No vocabulary correction table, OCR, CV,
extra model pass, or ADR change is authorized.
