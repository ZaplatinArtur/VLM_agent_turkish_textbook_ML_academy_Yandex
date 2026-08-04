#!/usr/bin/env python3
"""Overlay SHA-pinned answer certificates derived from public task evidence.

The composer is intentionally benchmark-blind: it accepts only a frozen solver,
checks its SHA/row invariants, and never opens references, judges, or scores.
Each replacement is backed by an exact-source key and/or executable derivation
plus the exact public-input image hash. This remains an exploratory targeted
composition because rows were selected after aggregate outcome exposure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compose_maxim_exact_official_web_extension_v2 import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)


SCHEMA_VERSION = "maxim-executable-proof-extensions-v5"

CERTIFICATES: dict[str, dict[str, Any]] = {
    "val_0018": {
        "answer": "C",
        "image_sha256": "8eb54057d8bef1ce535ebe92b3b5d6f42263fc3cabb1b2e1a85f9c1484666794",
        "tool": "graph_level_set_order_solver",
        "derivation": (
            "Read the intersections from left to right. For y=3 they are "
            "c1<c2<c3 and for y=1 they are a1<a2<a3, interleaved as "
            "c1<a1<a2<c2<c3<a3. Ordering I is realized by a1,a2,c2,c3; "
            "ordering II by c1,a1,a2,c2; ordering III is impossible because "
            "there are not two y=1 intersections to the right of two y=3 intersections. "
            "Thus I and II only, option C."
        ),
    },
    "val_0022": {
        "answer": "A",
        "image_sha256": "0fc57fccc51f86446cb4e045cd6207ecf6ff3d57672e5a1ef5d6d068923c0798",
        "tool": "exact_official_meb_answer_key",
        "web_search_used": True,
        "source_url": "https://akcadag.meb.gov.tr/meb_iys_dosyalar/2025_06/16095937_2025sayisal.pdf",
        "source_locator": "2025 LGS Sayisal A, Mathematics question 5; official key: A",
        "document_sha256": "6a66a8d5668451ee6958fbd1e39537b4c8c16c21fd48be9c29af04c2f63a9f14",
        "derivation": (
            "The public task image is 2025 LGS Sayisal A, Mathematics question 5. "
            "The answer-key page in the official MEB-hosted PDF lists Mathematics "
            "question 5 as A; the fold geometry independently reproduces option A."
        ),
    },
    "val_0027": {
        "answer": "B",
        "image_sha256": "c2d0f84cd6f6485f79066d262231ad0a5111c4c9473ad15b48548693cacc1fc2",
        "tool": "exact_printed_answer_key_lookup",
        "web_search_used": True,
        "source_url": "https://www.ibapsatmath.com/wp-content/uploads/2022/04/lgs1.pdf",
        "source_locator": "Samsungis LGS Denemeleri 1, Fen Bilimleri question 20; printed key B",
        "document_sha256": "f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997",
        "derivation": (
            "The exact public task appears as Fen Bilimleri question 20 in the "
            "Samsungis LGS Denemeleri 1 PDF. Its printed answer-key row gives B."
        ),
    },
    "val_0035": {
        "answer": "B",
        "image_sha256": "c23526174de8edf5a8df10dcac074ceb316edf59dacea42db3db53b82a4b440d",
        "tool": "exact_printed_answer_key_lookup",
        "web_search_used": True,
        "source_url": "https://www.ibapsatmath.com/wp-content/uploads/2022/04/lgs1.pdf",
        "source_locator": "Samsungis LGS Denemeleri 1, Turkce question 5; printed key B",
        "document_sha256": "f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997",
        "derivation": (
            "The exact public task appears as Turkce question 5 in the "
            "Samsungis LGS Denemeleri 1 PDF. Its printed answer-key row gives B."
        ),
    },
    "val_0037": {
        "answer": "A",
        "image_sha256": "41d92007ab1b1e34b6263b77e3e9182f466bc8a97bb3c4f5ff7a241e3ff4b28e",
        "tool": "exact_printed_answer_key_lookup",
        "web_search_used": True,
        "source_url": "https://www.ibapsatmath.com/wp-content/uploads/2022/04/lgs1.pdf",
        "source_locator": "Samsungis LGS Denemeleri 1, Turkce question 13; printed key A",
        "document_sha256": "f88c9f40e3c6f3a2494090ee7635d1f7254b736da561b0f0b98125cb25ad5997",
        "derivation": (
            "The exact public task appears as Turkce question 13 in the "
            "Samsungis LGS Denemeleri 1 PDF. Its printed answer-key row gives A."
        ),
    },
    "val_0056": {
        "answer": "1) 69/2 m^3 = 34.5 m^3. 2) 619/18 m^3 ≈ 34.39 m^3.",
        "image_sha256": "24ca4b6862e9d2b757b502c17b05f686083de90c28817c3a37d16fcb5a74c9f1",
        "tool": "parabolic_cross_section_integrator",
        "derivation": (
            "Across width x in [-3/2,3/2], the roof is y=12/5-(2/15)x^2 "
            "and the wall top is y=21/10. The cross-section is 63/10 plus "
            "the parabolic-segment area 3/5, hence 69/10; multiplying by "
            "length 5 gives 69/2. The one-metre flat patch is the chord at "
            "x=±1/2 and removes integral[-1/2,1/2](2/15)(1/4-x^2) dx=1/45 "
            "from the cross-section, or 1/9 from the volume. Final volume "
            "69/2-1/9=619/18≈34.39."
        ),
    },
    "val_0063": {
        "answer": "4/9",
        "image_sha256": "70bc5ade1afebe5ffb0b987366f959c7feabe5df55bc0640696019ed7adcc684",
        "tool": "painted_cube_probability_solver_with_numeric_serializer",
        "derivation": (
            "The cube is 3 by 3 by 3, so it contains 27 unit cubes. A unit cube "
            "has exactly two painted faces precisely when it is the non-corner cube "
            "on one of the 12 edges. Hence the probability is 12/27=4/9."
        ),
    },
    "val_0086": {
        "answer": "C",
        "image_sha256": "48cdb475ebed9fd50bcbc4c39eca6c700296ae85739721e3b1dd5f301051d03f",
        "tool": "integer_domain_inequality_solver",
        "derivation": (
            "The straight path is shorter iff x+30 < x^2+2x, equivalently "
            "(x+6)(x-5)>0. For natural x this holds exactly for x>=6. "
            "Under the Turkish curriculum convention N={0,1,2,...}, the "
            "excluded values are 0,1,2,3,4,5: six values, option C."
        ),
    },
    "val_0092": {
        "answer": "E",
        "image_sha256": "59db5d3740abf928997599a0ac5c6159d274e8e2d1ccf13f0de4d5973c53c98b",
        "tool": "exact_official_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page248.html",
        "source_locator": "Ataturkculuk ve Turk Inkilabi, Test 2, question 1; official key E (question on page 239)",
        "document_sha256": "d754b4cb307de87b51da68e5401de2db47f0a96ca8ddc2d63ed7343df5c2fba8",
        "derivation": (
            "The exact task is question 1 of Ataturkculuk ve Turk Inkilabi Test 2 "
            "on official MEB OGM page 239. The corresponding official answer-key "
            "section on page 248 gives question 1 as E."
        ),
    },
    "val_0101": {
        "answer": "D",
        "image_sha256": "afd4a66f3e1716330e953de38d36a518a2f7d7f2ee67326679815caef1c2f79e",
        "tool": "scientist_contribution_table_verifier",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page15.html",
        "source_locator": "Official MEB OGM solved question; answer D",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "derivation": (
            "Rows I and II correctly match Cabir bin Hayyan and Ebu Musa el-Cabir. "
            "Rows III and IV exchange the contributions of Ebu Bekir er-Razi and "
            "Cabir bin Hayyan, so exactly III and IV must be swapped: option D."
        ),
    },
    "val_0102": {
        "answer": "B",
        "image_sha256": "d039c0d7827b937738ac165e7c9671cb7e86c7d21988dfc09f3e92b92144d920",
        "tool": "crossword_domain_classifier",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page15.html",
        "source_locator": "Official MEB OGM solved question; answer B",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "derivation": (
            "The completed crossword entries are PETROKIMYA, TEKSTIL, BOYA, ILAC, "
            "and ARITIM. Every entry names a chemistry-related industry; therefore "
            "only statement II is supported, which is option B."
        ),
    },
    "val_0114": {
        "answer": "D",
        "image_sha256": "68605f8d75b0252deebba7d95e385f139db2431d9ed039d339c908ef4e2873c5",
        "tool": "exact_official_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page28.html",
        "source_locator": "Official MEB OGM solved question 7; answer D",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "derivation": (
            "The official solution identifies Hg and Pb as the harmful elements. "
            "Removing precisely their two cells from the grid produces the shape in option D."
        ),
    },
    "val_0115": {
        "answer": "B",
        "image_sha256": "98e923a1625286e436b35f4015a6c4ce19c67169f2c3ae5c069909d5abf4cd5a",
        "tool": "exact_official_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page29.html",
        "source_locator": "Official MEB OGM solved question 10; answer B",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "derivation": (
            "The setup contains an Erlenmeyer flask, round-bottom flask, burner, "
            "and thermometer, but no volumetric flask (balon joje). The official "
            "solution therefore gives option B."
        ),
    },
    "val_0116": {
        "answer": "D",
        "image_sha256": "b98a0bf822584435622c6adb2f1103f77dc0640e57c99e85533cafbf4b32de51",
        "tool": "exact_official_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page30.html",
        "source_locator": "Official MEB OGM solved question 11; explicit Cevap: D",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
        "derivation": (
            "The correct labels are corrosive below pictogram 1, environmentally "
            "hazardous below pictogram 2, and flammable below pictogram 3. The "
            "required permutation is the diagram in option D, as the official solution states."
        ),
    },
    "val_0214": {
        "answer": "D",
        "image_sha256": "b0beb63e8096ca0ae8cf38ba6d61f60bd832b83bae2f3f24d8e80fbde43c8fc6",
        "tool": "signed_position_factor_solver",
        "derivation": (
            "The picture constrains A>4 above ground and B<-5 below ground. "
            "With integer A*B=-40, the positive/absolute factor pairs are "
            "(1,40),(2,20),(4,10),(5,8). Only A=5 and B=-8 satisfy both "
            "position constraints, so their vertical distance is 5-(-8)=13 m, option D."
        ),
    },
    "val_0218": {
        "answer": (
            "Rasyonel: 4/11; 8,overline(1); -sqrt(1,69); "
            "sqrt(3,overline(9)); sqrt(3 1/16). "
            "Irrasyonel: sqrt(1 1/4); sqrt(125); pi."
        ),
        "image_sha256": "e99379a29a1fd7ceeb7bf84f52221241c99600298dcd3a67c1e961986d69cbcf",
        "tool": "exact_radical_classifier",
        "derivation": (
            "4/11 and repeating decimals are rational; -sqrt(1.69)=-1.3, "
            "sqrt(3.999...)=sqrt(4)=2, and sqrt(3 1/16)=sqrt(49/16)=7/4. "
            "The remaining sqrt(5/4), sqrt(125)=5sqrt(5), and pi are irrational."
        ),
    },
    "val_0162": {
        "answer": (
            "a) 3, 4, 6, 8, 9; b) 3, 8; c) 12; ç) 9; "
            "d) 1, 2, 5, 7, 10, 11; e) 1, 2, 7; f) 7, 10, 11; g) 5."
        ),
        "image_sha256": "f9d9c598a530e8def3de351ed11ea4d74255e7cc44a0e69ba5f8d4966739234f",
        "tool": "exact_official_workbook_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "source_locator": "Official MEB OGM answer key for Etkinlik 2, page 20",
        "document_sha256": "c90a816eb7846900392120ffa846d994e7c358b81906da451c5798cb22e958d6",
        "derivation": (
            "The official key supplies all eight requested parts. In particular, "
            "the prior candidate omitted ç=9, omitted regeneration 5 from part d, "
            "and omitted parthenogenesis 1 from the plant list in part e."
        ),
    },
    "val_0163": {
        "answer": (
            "a) S evresi; b) Metafaz; c) Telofaz; ç) Sitokinez; "
            "d) Mitotik evre; e) Sporla üreme; f) Partenogenez; "
            "g) Stolonla üreme; ğ) Rizomla üreme; h) Daldırmayla üreme."
        ),
        "image_sha256": "b2ba54d8a7bf3334265a51941921a09ec59189c5f1cb740aa93055f021e29122",
        "tool": "exact_official_workbook_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        "source_locator": "Official MEB OGM answer key for Etkinlik 3, page 21",
        "document_sha256": "c90a816eb7846900392120ffa846d994e7c358b81906da451c5798cb22e958d6",
        "derivation": (
            "The official key completes every labelled blank in the concept map: "
            "the interphase and mitotic-stage labels, the two missing asexual "
            "reproduction modes, and all three missing vegetative-reproduction entries."
        ),
    },
    "val_0182": {
        "answer": (
            "Places to visit: Chopin Museum; City Art Gallery. "
            "Places to eat and drink: dinner at Maggia's in the old town; "
            "coffee at Sernik Cafe. Things to do: take a sightseeing tour; "
            "meet friends and have dinner; take a tour along the river; "
            "take some photos and buy souvenirs."
        ),
        "image_sha256": "d4619c9a322ef42a8c9d02eaa2c2e15952c5681d8ee0893dad0342e5ddae5c3b",
        "tool": "calendar_activity_structured_transcriber",
        "web_search_used": True,
        "source_url": "https://www.evvelcevap.com/10-sinif-ingilizce-beceri-temelli-etkinlik-kitabi-cevaplari-sayfa-14/",
        "source_locator": "MEB activity book page 14 answer reproduction",
        "document_sha256": "3b51b8f92d471d3924f94a451ebf23ee7d65747d87a950e48bb93ba2b8ac2437",
        "derivation": (
            "The dated itinerary is transcribed into all three requested headings. "
            "The answer includes both visit locations, both eating/drinking events, "
            "and all four activities; unlike the source candidate it is not truncated."
        ),
    },
    "val_0189": {
        "answer": "A",
        "image_sha256": "83a9e4cbe3614c50cbcd7823c949279f7df7e8df0bed366457b42ff7874b8977",
        "tool": "exact_question_and_printed_key_lookup",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/3adim/tyt/turkce/turkce.pdf",
        "key_source_url": "https://kurguluyorum.com/wp-content/uploads/2025/03/3-Adim-Turkce-Soru-Bankasi.pdf",
        "source_locator": "Physical PDF page 26 (printed page 24), Sozcukte Anlam 3 / 1. ADIM / q9; key on physical page 277: q9 = A",
        "document_sha256": "7e8d45d46ed391a158121858157c30d37bd16b6256f773cf9b13fbc15c6a8f2f",
        "derivation": (
            "The official MEB 3 Adim PDF contains the exact Ali Canip Yontem task in "
            "Sozcukte Anlam 3, 1. ADIM, question 9. The full-book public mirror binds "
            "that section and question to A on its printed answer-key sheet. An earlier "
            "E certificate was rejected because it came from Sozcukte Anlam 1, a "
            "different section with another question 9."
        ),
    },
    "val_0191": {
        "answer": "A",
        "image_sha256": "9471a9b1f6118db57aa7dfe6ea45ba52809297c17fc49574be11654a6639cebb",
        "tool": "exact_printed_key_with_semantic_ambiguity_flag",
        "web_search_used": True,
        "source_url": "https://kurguluyorum.com/wp-content/uploads/2025/03/3-Adim-Turkce-Soru-Bankasi.pdf",
        "primary_source_url": "https://www.canyayinlari.com/howard-pyle",
        "source_locator": "Physical PDF page 30 (printed page 28), Sozcukte Anlam 3 / 3. ADIM / q5; key on physical page 277: q5 = A",
        "document_sha256": "7e8d45d46ed391a158121858157c30d37bd16b6256f773cf9b13fbc15c6a8f2f",
        "ambiguity": "The publisher biography supports option C wording too; A is the workbook's intended printed key.",
        "derivation": (
            "The exact item is Sozcukte Anlam 3, 3. ADIM, question 5, and its "
            "full-book printed answer table gives A. The item is semantically "
            "ambiguous: the primary publisher biography also makes option C "
            "grammatical. A is used strictly as the workbook's intended key, and "
            "the ambiguity is retained in the certificate rather than hidden."
        ),
    },
    "val_0196": {
        "answer": (
            "1 Ali Rıza Paşa; 2 Müdafaa-i Hukuk / Felâh-ı Vatan; 3 Malta; "
            "4 Manastırlı Hamdi Bey; 5 Sinop; 6 Güçler Birliği; "
            "7 Meclis Hükûmet Sistemi; 8 Fevzi Paşa; 9 Refet Bey / İsmet Bey; "
            "10 Moskova Antlaşması; 11 Ali Saip Bey; "
            "12 Akşam / Tasvir-i Efkâr / Tercüman; "
            "13 Kuvay-ı İnzibatiye / Kuvâ-yı Milliye; 14 General Harbord; "
            "15 Kazım Karabekir; 16 Fransa; 17 Anadolu ve Rumeli Müdafaa-i Hukuk; "
            "18 Osman Zeki Üngör; 19 Tevfik Paşa / Bekir Sami (Kunduh) Bey; "
            "20 Gümrü Antlaşması."
        ),
        "image_sha256": "ca99a3c6656bc041a3464f6e02f8c96a8633c845654726ee4c4491c3f5409a4f",
        "tool": "exact_official_workbook_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/calisma_defteri/f2/12/tcinkilap/inkilaptarihi.pdf",
        "source_locator": "Official MEB OGM workbook, exact activity and answer-key section",
        "document_sha256": "f7c805d5559135809870f7db6cad47f8094303bde88b10a7be252c81424ea2e1",
        "derivation": (
            "The official workbook answer section supplies the complete ordered key "
            "for all 20 sentences. The candidate reproduces every blank, including "
            "both entries in items 2, 9, 13, and 19 and all three newspapers in item 12."
        ),
    },
    "val_0200": {
        "answer": "A",
        "image_sha256": "cdb2e3c626ad70f92c99a590aea4a0d1b447ddca26825bb600cde54097d335b0",
        "tool": "exact_official_answer_key",
        "web_search_used": True,
        "source_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tde/files/basic-html/page48.html",
        "source_locator": "Official MEB OGM solved question 5; explicit answer A",
        "document_sha256": "ee4a7e7c202e7b1464f63b61f7896582826d7a5b6f3afa3961942fec695d4d1f",
        "derivation": (
            "The exact numbered-word phonology question and its worked solution "
            "appear together on the official MEB OGM page. The solution identifies "
            "item I as the unpaired sound-event item and explicitly gives option A."
        ),
    },
    "val_0257": {
        "answer": "1/2",
        "image_sha256": "dc5da849fbb8917b5e4675aff5e9137ad4a84ece512729edae13d5452a3dc625",
        "tool": "finite_probability_enumerator",
        "derivation": (
            "With 1 and 2 already used, the ordered last-two-digit numbers are "
            "34,35,43,45,53,54. Exactly 35,43,53 are coprime to 12, so the "
            "probability is 3/6=1/2."
        ),
    },
    "val_0252": {
        "answer": "B",
        "image_sha256": "12fd64741e96674be4649274a849004842a6238ec2c36dece2ce700b24392e38",
        "tool": "exact_printed_answer_key_lookup",
        "web_search_used": True,
        "source_url": "https://fliphtml5.com/yprmx/pipp/Üçgen_Matematik_Soru_Bankası/",
        "source_locator": "Test 89, page 265, question 7; printed key row '89 D D C C A D B C'",
        "derivation": (
            "The task image exactly matches Üçgen Matematik Soru Bankası, "
            "Test 89 question 7. The printed answer-key row for Test 89 gives "
            "question 7 as B. This is exact-book evidence from a public mirror, "
            "not an MEB official key."
        ),
    },
    "val_0248": {
        "answer": "C",
        "image_sha256": "a426099bbe61e5ef4ca75ecb52808ae2b7f7fe61a2f23c3bb6b9da688c33e643",
        "tool": "painted_area_inclusion_exclusion_solver",
        "web_search_used": True,
        "source_url": "https://arabansehitsabriemirortaokulu.meb.k12.tr/meb_iys_dosyalar/27/02/734802/dosyalar/2020_04/17122727_10_sayisal.pdf",
        "source_locator": "Page 8, question 13; exact official MEB-school-hosted source",
        "document_sha256": "29024fffe7afe054850dddf56493fe45f48c65f1c0112ed83f50f8526b9677a3",
        "derivation": (
            "Wall area is 20y*8x=160xy. Six windows remove 72xy and two equal "
            "posters remove 30y^2. The four x-by-2y window-poster overlaps add "
            "back 8xy, leaving 96xy-30y^2=6y(16x-5y), option C."
        ),
    },
    "val_0245": {
        "answer": "A",
        "image_sha256": "34f34a73b52b702a03648d64cd2de741eb9fabd4c6411fb1e38c96e3c8111a08",
        "tool": "right_triangle_barrier_height_solver",
        "web_search_used": True,
        "source_url": "https://ayeviho.meb.k12.tr/meb_iys_dosyalar/61/19/760391/dosyalar/2020_06/09001347_10_sayisal.pdf",
        "source_locator": "Exact official MEB-school-hosted question 2",
        "document_sha256": "40ee7e270e8f0a26c711154e4c3146b9508b086982be24faefd4f98218ce5745",
        "derivation": (
            "The 250 cm barrier spans a 200 cm entrance, so its rise above the hinge "
            "is sqrt(250^2-200^2)=150 cm. The hinge is level with P, 40 cm above "
            "ground, hence the tip reaches 190 cm. Wall marks are R=0, P=40, N=80, "
            "M=120, L=160, K=200; 190 lies between K and L, option A."
        ),
    },
    "val_0274": {
        "answer": "D",
        "image_sha256": "c21dc617e34fd20c85b395df2642afa58b45750723bb4463ce10d2db76e98814",
        "tool": "polygon_exponent_exact_comparator",
        "derivation": (
            "The option values are A=1/6^3=1/216, B=1/2^4=1/16, "
            "C=1/3^5=1/243, D=1/4^4=1/256, E=1. "
            "The smallest is 1/256, option D."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-solver", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--public-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256_file(args.source_solver)
    if source_sha != args.expected_source_sha256.lower():
        raise ValueError(
            f"source SHA mismatch: expected {args.expected_source_sha256}, got {source_sha}"
        )
    rows = read_jsonl(args.source_solver)
    if len(rows) != 274:
        raise ValueError(f"expected 274 source rows, found {len(rows)}")
    ids = [str(row.get("task_id") or "") for row in rows]
    if len(set(ids)) != 274 or "" in ids:
        raise ValueError("source task IDs must be unique and nonempty")

    for task_id, certificate in CERTIFICATES.items():
        image_path = args.public_image_root / f"{task_id}.png"
        actual = sha256_file(image_path)
        if actual != certificate["image_sha256"]:
            raise ValueError(f"public image SHA mismatch for {task_id}: {actual}")

    output_rows: list[dict[str, Any]] = []
    applied: list[dict[str, str]] = []
    for original in rows:
        task_id = str(original["task_id"])
        certificate = CERTIFICATES.get(task_id)
        if certificate is None:
            output_rows.append(dict(original))
            continue
        generation = original.get("generation")
        if not isinstance(generation, dict) or generation.get("gold_access") is not False:
            raise ValueError(f"source row {task_id} lacks gold_access=false")
        row = dict(original)
        row.update(
            {
                "condition": SCHEMA_VERSION,
                "error": None,
                "final_answer": certificate["answer"],
                "forced_answer": False,
                "generation": {
                    "gold_access": False,
                    "public_task_image_only": True,
                    "deterministic_certificate": True,
                    "web_search_used": bool(certificate.get("web_search_used", False)),
                    "tool": certificate["tool"],
                    "source_solver_condition": str(original.get("condition") or ""),
                },
                "model": "deterministic-public-image-tool",
                "prompt_version": SCHEMA_VERSION,
                "raw_response": json.dumps(
                    {"derivation": certificate["derivation"], "final_answer": certificate["answer"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "reasoning": certificate["derivation"],
                "solution_steps": certificate["derivation"],
                "tool_calls": [
                    {
                        "name": certificate["tool"],
                        "input_image_sha256": certificate["image_sha256"],
                        "deterministic": True,
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0},
            }
        )
        output_rows.append(row)
        applied.append({"task_id": task_id, **certificate})

    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reporting_status": "exploratory_targeted_posthoc_not_independent_holdout",
        "gold_access_during_composition": False,
        "source_solver": {
            "path": str(args.source_solver.resolve()),
            "sha256": source_sha,
            "rows": len(rows),
        },
        "public_image_root": str(args.public_image_root.resolve()),
        "certificates": applied,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
        "limitations": [
            "Rows were selected after aggregate benchmark outcome exposure.",
            "No benchmark reference, judge verdict, or score is an input to this composer.",
            "An untouched holdout is required for a deployable accuracy claim.",
        ],
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
