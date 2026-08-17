# Amendment: token evidence and stable region identity

The convergence scorer must construct candidates from partial token and phrase
agreement rather than requiring whole-string equality. It must preserve scoring
basis in JSON, support lexical variants deterministically, and keep unresolved
tokens only where evidence is insufficient. Experimental region IDs must retain
their original localization indexes across duplicate rejection so reread and
TrOCR comparison artifacts agree (for example, `region-10`). No model inference
or specimen-specific correction is permitted.
