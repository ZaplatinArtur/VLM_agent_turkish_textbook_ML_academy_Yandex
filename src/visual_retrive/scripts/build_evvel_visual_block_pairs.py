"""Build SigLIP pairs from block-level OpenRouter visual query generation."""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, io, json, re
from collections import Counter, defaultdict
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def subject(row):
    text = " ".join(str(row.get(k) or "") for k in ("book_slug", "title", "page_url")).casefold()
    mapping = [("matematik","math"),("fizik","physics"),("kimya","chemistry"),("biyoloji","biology"),
               ("tarih","history"),("cografya","geography"),("coğrafya","geography"),("edebiyat","turkish language and literature"),
               ("ingilizce","english"),("almanca","german"),("din-kulturu","religious culture and ethics"),
               ("felsefe","philosophy"),("psikoloji","psychology"),("sosyoloji","sociology"),("mantik","logic")]
    return next((value for needle,value in mapping if needle in text), "high school other")


def slug(row):
    value=f"evvel-{int(row['grade'])}-sinif-{row['book_slug']}".casefold()
    return re.sub(r"[^a-z0-9_-]+","-",value).strip("-")


def render_text(text):
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",25)
    probe=ImageDraw.Draw(Image.new("RGB",(1,1))); width=1400; margin=60; lines=[]
    for paragraph in (text or "").splitlines():
        current=""
        for word in paragraph.split():
            candidate=(current+" "+word).strip()
            if probe.textlength(candidate,font=font)<=width-2*margin: current=candidate
            else:
                if current: lines.append(current)
                current=word
        if current: lines.append(current)
    lines=lines[:90]; image=Image.new("RGB",(width,max(400,120+34*len(lines))),"white"); draw=ImageDraw.Draw(image)
    for i,line in enumerate(lines): draw.text((margin,margin+i*34),line,font=font,fill="black")
    return image


def materialize(row, root):
    book=slug(row); digest=hashlib.sha1(str(row["block_id"]).encode()).hexdigest()[:16]
    numeric_page=int(digest,16)
    rel=Path("books")/book/"pages"/f"{numeric_page:04d}.jpg"; target=root/rel
    if target.is_file(): return rel.as_posix(),numeric_page
    target.parent.mkdir(parents=True,exist_ok=True)
    if row.get("source_type")=="rendered_text" and row.get("source_text"):
        image=render_text(row["source_text"])
    else:
        response=requests.get(row["image_url"],timeout=(15,90),headers={"User-Agent":"TurkishTextbookResearchBot/1.0"})
        response.raise_for_status(); image=Image.open(io.BytesIO(response.content)).convert("RGB")
    temp=target.with_suffix(".part"); image.save(temp,"JPEG",quality=90,optimize=True); temp.replace(target)
    return rel.as_posix(),numeric_page


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args(); rows=[json.loads(x) for x in a.input.open(encoding="utf-8") if x.strip()]
    built=[]; failed=0
    def build(row):
        rel,numeric_page=materialize(row,a.data_root); page_id=f"{slug(row)}:{numeric_page}"
        return [{"query":query,"positive_page_id":page_id,"positive_image":rel,"positive_answer_text":row.get("source_text", ""),
                 "hard_negative_page_ids":[],"same_source_page_ids":[page_id],"subject":subject(row),"grade":int(row["grade"]),
                 "book_slug":slug(row),"source":"evvelcevap:openrouter_visual_block","source_url":row["page_url"]}
                for query in row.get("synthetic_queries") or []]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
      futures={executor.submit(build,row):row for row in rows}
      for i,future in enumerate(concurrent.futures.as_completed(futures),1):
        try:
            built.extend(future.result())
        except Exception as exc:
            failed+=1
            if failed<=20: print(f"block_failed={type(exc).__name__}:{exc}",flush=True)
        if i%100==0 or i==len(rows): print(json.dumps({"blocks_processed":i,"blocks_total":len(rows),"pairs":len(built),"failed":failed}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8") as out:
        for row in built: out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"blocks":len(rows),"pairs":len(built),"failed":failed,"grades":Counter(x["grade"] for x in built)},ensure_ascii=False,default=dict),flush=True)

if __name__=="__main__": main()
