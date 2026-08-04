# Remaining public-evidence audit v1

> **Status:** post-hoc diagnostic only. This does not replace the frozen standard metric.

## Result

- Frozen standard metric (unchanged): **263/274 = 0.959854**.
- Independently evidence-confirmed standard-score disagreements: **10**.
- Malformed/missing-prompt public payloads: **1**.
- Evidence-adjusted diagnostic, fixed denominator: **273/274 = 0.996350**.
- Evidence-adjusted diagnostic, answerable only: **273/273 = 1.000000**.

## Evidence certificates

| Task | Candidate | Tier | Public source | Proof | Image SHA-256 |
|---|---:|:---:|---|---|---|
| val_0063 | 4/9 | B | 3 x 3 x 3 painted-cube diagram | There are 27 unit cubes. Exactly two painted faces occur only on the non-corner cube of each of the 12 edges, so P=12/27=4/9. | `70bc5ade1afebe5ffb0b987366f959c7feabe5df55bc0640696019ed7adcc684` |
| val_0073 | B | B | visible multiplication 243 x [two digits] = 18225 | 18225/243=75; the requested digit product is 7*5=35, option B. | `1b2f6cd9f8dd36d721e13d4a85e96b5f59608bdb490fbaf7a4a5e3c10d3a13cd` |
| val_0076 | C | B | 40 cm by 24 cm rectangle | A 25% reduction gives side lengths 30 and 18; 2*(30+18)=96 cm, option C. | `04afaa0b0e4af5bf806806c1e29fab61139ffa5ac1d14472bbcec6b31a4b90be` |
| val_0165 | C | B | five labelled mitosis-stage drawings | The drawings are IV interphase, I prophase, V metaphase, III anaphase, II telophase/cytokinesis. Thus IV-I-V-III-II, option C. | `7826d9b4af5c3e3c59bdff04d9da8d6e4f6a5dabccb4c549231073598238ef65` |
| val_0170 | A | A | [MEB Defterim Biyoloji 10, question 32 and printed key 32.A](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page38.html) | The exact official public question is numbered 32 and the official printed key records 32.A; biologically, only bacteria in the shown set reproduce exclusively asexually. | `644191a566296f651a3b1d7b9e902b1d913aa02125198fea8f83fbfe1fc4602e` |
| val_0186 | B | B | complete sociology passage and options | The passage says legal rules emerged when increasing social complexity made prior rules insufficient. Law therefore responds to social needs, option B. | `e2aca152e93d0d2ac63e194839e1e9e22bd81d32e0975d3682910e4bb04d553b` |
| val_0208 | Naneli: 12 kutu, Limonlu: 8 kutu | B | complete mint/lemon box word problem | Equal mass gives 4m=6l, so m=3k and l=2k. Revenue gives 45(3k)+60(2k)=255k=1020, hence k=4: 12 mint and 8 lemon boxes. | `f72b3f24114007d7e52385886db2ed512c98c88cc54ec0a3565f7af829a1e3bf` |
| val_0243 | A | B | 107-book power-capacity packing problem | With one box of each type, three positive powers sum even, so three boxes cannot total odd 107. Four work: 3^4+2^4+5^1+5^1=81+16+5+5=107. Minimum 4, option A. | `a79f3da421f4ab6c1510890d6dd64e0bc1b297bd2aebb76780fd0b8bf53825b6` |
| val_0251 | A | B | quadrilateral with sides BP, 9, 11, 5 | For integer BP=1 the sides 1,5,9,11 satisfy the non-degenerate quadrilateral inequality 11<1+5+9. Therefore the minimum positive integer is 1, option A. | `d07c49f6436aecda798768b86e3e7dd6615e1aaeda5a87f7ac93e1c18d3b1195` |
| val_0257 | 1/2 | B | finite password probability problem | The possible ordered final pairs are 34,35,43,45,53,54. Exactly 35,43,53 are coprime to 12, so the probability is 3/6=1/2. | `dc5da849fbb8917b5e4675aff5e9137ad4a84ece512729edae13d5452a3dc625` |

## Malformed public payload

| Task | Finding | Image SHA-256 |
|---|---|---|
| val_0100 | The image is a Yandex street advertisement/photo and contains no chemistry question, answer options, or answerable task prompt. | `b781b114b9485c10cd49ceeca3a4f6ff6302e5f9c72c8aa6a9996c2e4dd5f9bb` |

## SHA lineage and isolation

- Solver: `6544b16aee4c6d09067a5ec8fb405de9053f7b85d6c45392713ccbcc73f8875d` (274 rows).
- Frozen standard score: `7e4861144e21456304ba1d1ff06811172637e379a3482badaaa0bcbb4d5c20f3` (274 outcomes).
- Certificate registry: `81b56414122cccd75a004111c8fb6b8e89d18b85645ab1d94241e31448747302`.
- Audit utility: `1d4026cc7f0ee4d8f031bae9d0c6a68d217819061c203077440ca7c79f1f9544`.
- No benchmark/reference/judge/gold input was accepted; score provenance paths were not followed.

## Limitations

- This is a post-hoc disagreement audit, not a preregistered or blind evaluation.
- The frozen standard score is unchanged; evidence-adjusted values are diagnostic only.
- Tier C certificates depend on contextual linguistic interpretation rather than an explicit official answer key.
- The malformed-row exclusion is shown only as an answerable-only diagnostic; the fixed-denominator view retains all 274 rows.
- No benchmark, reference answer, judge file, gold field, network service, or GPU was consulted by this audit utility.
