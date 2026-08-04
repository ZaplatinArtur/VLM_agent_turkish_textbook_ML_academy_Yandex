#!/usr/bin/env python3
"""Overlay ten independently frozen exact official MEB/OGM certificates.

The four v2 certificates are extended by six exact solved-page/key matches.
This entry point delegates composition to v2 after replacing its immutable
certificate map; neither entry point accepts evaluation or gold inputs.
"""

from __future__ import annotations

import compose_maxim_exact_official_web_extension_v2 as base


EXTRA_OVERRIDES: dict[str, dict[str, str]] = {
    "val_0087": {
        "answer": "B",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page211.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page211.html",
        "key_locator": "Solved question 1; explicit Cevap:B on the official page",
        "document_sha256": "d754b4cb307de87b51da68e5401de2db47f0a96ca8ddc2d63ed7343df5c2fba8",
    },
    "val_0088": {
        "answer": "B",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page212.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tarih/files/basic-html/page212.html",
        "key_locator": "Solved question 4; explicit Cevap:B on the official page",
        "document_sha256": "d754b4cb307de87b51da68e5401de2db47f0a96ca8ddc2d63ed7343df5c2fba8",
    },
    "val_0094": {
        "answer": "D",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tde/files/basic-html/page14.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/tde/files/basic-html/page14.html",
        "key_locator": "Solved question 11; explicit Cevap:D on the official page",
        "document_sha256": "ee4a7e7c202e7b1464f63b61f7896582826d7a5b6f3afa3961942fec695d4d1f",
    },
    "val_0123": {
        "answer": "B",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page38.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/kimya/files/basic-html/page38.html",
        "key_locator": "Solved question 41; explicit Cevap:B on the official page",
        "document_sha256": "b3b89a296d91ec0c3c7d6862d3113e0b7dcf34469c993808715203bbf0df1e7f",
    },
    "val_0141": {
        "answer": "D",
        "authority": "MEB OGM Materyal",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page26.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/konu-pekistirme/tyt/fizik/files/basic-html/page26.html",
        "key_locator": "Solved question 15; explicit Cevap:D on the official page",
        "document_sha256": "7254325f6a477b745782566d3281af03d3f153af2e2c4f2cf3ae8f83f4388480",
    },
    "val_0179": {
        "answer": "B",
        "authority": "MEB Defterim Biyoloji 10",
        "question_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page92.html",
        "key_url": "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page180.html",
        "key_locator": "Unit 2 end evaluation, Test 5, question 19; key 19.B",
        "document_sha256": "640bb362f2d53d31663326ac303c5065f4670f2a0d506300beb5e41869384e2b",
    },
}


base.SCHEMA_VERSION = "maxim-exact-official-web-extension-composition-v3"
base.OVERRIDES = {**base.OVERRIDES, **EXTRA_OVERRIDES}


if __name__ == "__main__":
    raise SystemExit(base.main())
