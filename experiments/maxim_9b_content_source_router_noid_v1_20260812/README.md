# Maxim274 content/source router without ID routing

This experiment separates two claims:

- Arm A overlays new content rules on the archived 249/274 artifact. It is a
  diagnostic only because the archived base already contains exact-ID
  component composition.
- Arm B starts from archived base240 and chooses every replacement from OCR
  content and official-source evidence only.

The selection function accepts `ocr_text`, `answer_type`, and `input_mode`.
It has no parameter for benchmark/task ID, input filename, image/content SHA,
reference, prior verdict, or score. IDs enter only after selection to align
the chosen answer with the output schema. The frozen decision file is
identity-free and supports a counterfactual re-identification audit.

No model/API/GPU call is made. `build` creates both candidates and their
freeze; scoring is intentionally external and must happen only afterwards.
