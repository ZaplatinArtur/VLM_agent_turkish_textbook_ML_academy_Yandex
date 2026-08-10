# Targeted exact official-web evidence

Status: exploratory targeted posthoc evidence, not an independent holdout result.

## Measured result

The deterministic exact-web branch measured **210/274 = 0.766423**. Relative to the current strict frozen branch at 205/274 = 0.748175, it produced **5 fixes and 0 regressions**.

Only six deterministic-score rows were substituted: `val_0003`, `val_0110`, `val_0131`, `val_0170`, `val_0173`, and `val_0194`. Five became measured fixes. `val_0170=A` remained incorrect under the frozen deterministic scorer even though the exact MEB question and printed key say A; it is recorded as a benchmark-reference conflict rather than counted as a fix.

The other five researched rows use the frozen image-judge partition. They were deliberately not substituted, preserving compatibility with the existing image-judge artifact.

## Exact evidence rows

| Task | Answer | Evidence | Existing frozen-source status | Measured status |
|---|---:|---|---|---|
| `val_0003` | C | [Official OSYM 2023 AYT booklet and key](https://dokuman.osym.gov.tr/pdfdokuman/2023/YKS/TSK/yks_ayt_2023_kitapcik_g5A2H.pdf), question 17 / key 17.C; [official landing page](https://www.osym.gov.tr/2023yks-tyt-ayt-ve-ydt-temel-soru-kitapciklari-ve-cevap-anahtarlari) | Absent | Fixed |
| `val_0087` | **B** | [MEB OGM History page 211](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page211.html) | Absent | Image-judge row; not substituted |
| `val_0088` | B | [MEB OGM History page 212](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page212.html) | Absent | Image-judge row; not substituted |
| `val_0094` | D | [MEB OGM TDE page 14](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tde/files/basic-html/page14.html) | Present in 2 sources, all scored false | Image-judge row; not substituted |
| `val_0110` | D | [MEB OGM Chemistry question](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page26.html), [official key](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page321.html), 4.TEST q8 / 8.D | Absent | Fixed |
| `val_0123` | B | [MEB OGM Chemistry page 38](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page38.html) | Absent | Image-judge row; not substituted |
| `val_0131` | C | [MEB OGM Physics question](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page13.html), [official key](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page363.html), 1.TEST q4 / 4.C | Absent | Fixed |
| `val_0141` | D | [MEB OGM Physics page 26](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page26.html) | Present in 26 sources, all scored false | Image-judge row; not substituted |
| `val_0170` | A | [MEB Biology question 32](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page38.html), [printed key 32.A](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html) | Present in 36 sources, all scored false | Frozen deterministic-scorer conflict |
| `val_0173` | D | [MEB Biology question](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page45.html), [printed key](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html), Test 4 q1 / 1.D | Absent | Fixed |
| `val_0194` | A | [Official MEB PDF endpoint](https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/3adim/tyt/turkce/turkce.pdf) was unavailable; the [identical third-party copy](https://kurguluyorum.com/wp-content/uploads/2025/03/3-Adim-Turkce-Soru-Bankasi.pdf) has Yardimci Dusunce, 2.ADIM, q3 / 3.A | Absent | Fixed; lower source authority |

Important column warning for `val_0087`: the answer is **B**. Its own question-1 solution ends with `Cevap: B`. The nearby `Cevap: C` appears in the adjacent column and belongs to a different question.

## Projections, not measured scores

| Scenario | Result | Meaning |
|---|---:|---|
| Existing 37-final-source oracle (default + 36 alternatives) | 244/274 = 0.890511 | Reference oracle before these exact-web additions |
| Token-coverage projection | 252/274 = 0.919708 | 244 + eight answers absent from every existing source |
| Full adjudicated projection | 255/274 = 0.930657 | 244 + all eleven rows, including three scorer-conflict corrections |
| Strict baseline + new tokens | 213/274 = 0.777372 | 205 + the eight previously absent answers |
| Strict baseline + all adjudicated rows | 216/274 = 0.788321 | 205 + all eleven after conflict resolution |

None of the values above 0.90 is an achieved benchmark result. They are oracle/candidate-coverage projections derived after outcome exposure and cannot be presented as a production metric. A preregistered selector and untouched holdout are still required.

## Resource audit

Web search was used to locate and verify the public question/key pages. The exact-key branch required no new model generation, no shared GPU, and no external compute. Composition and scoring were local. The local Math certificate run was a separate CPU-only experiment and is not part of this web-evidence score.
