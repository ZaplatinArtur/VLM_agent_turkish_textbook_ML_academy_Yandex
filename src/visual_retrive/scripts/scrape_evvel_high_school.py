"""Scrape Evvel Cevap 9-12 textbook/page answer metadata without fake images."""
from __future__ import annotations
import argparse, concurrent.futures, html, json, re, time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests

BASE="https://www.evvelcevap.com"
PAGE_RE=re.compile(r"(?P<grade>9|10|11|12)-sinif-(?P<book>.+?)-sayfa-(?P<page>\d+)-cevabi(?:-\d+)?/?$")
SITEMAP_LOC_RE=re.compile(r"<loc>(.*?)</loc>",re.I|re.S)
IMG_RE=re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>",re.I|re.S)

class Parser(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.article=[]; self.main=[]; self.article_depth=0; self.main_depth=0; self.ignore_depth=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in {'script','style','noscript','svg'}: self.ignore_depth+=1
        if tag=='a' and a.get('href'): self.links.append(a['href'])
        if tag=='article': self.article_depth+=1
        if tag=='main': self.main_depth+=1
    def handle_endtag(self,tag):
        if tag=='article' and self.article_depth: self.article_depth-=1
        if tag=='main' and self.main_depth: self.main_depth-=1
        if tag in {'script','style','noscript','svg'} and self.ignore_depth: self.ignore_depth-=1
    def handle_data(self,data):
        if self.ignore_depth: return
        if self.article_depth and data.strip(): self.article.append(data.strip())
        elif self.main_depth and data.strip(): self.main.append(data.strip())

def fetch(session,url,retries=3):
    for attempt in range(retries):
        try:
            r=session.get(url,timeout=35); r.raise_for_status(); return r.content.decode('utf-8',errors='replace')
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in {404,410}: raise
            if attempt+1==retries: raise
            time.sleep(1.5*(attempt+1))
        except Exception:
            if attempt+1==retries: raise
            time.sleep(1.5*(attempt+1))

def parse_html(body):
    p=Parser(); p.feed(body); return p

def canonical(url):
    u=urljoin(BASE,url); parts=urlparse(u); return f"{parts.scheme}://{parts.netloc}{parts.path}"

def discover(max_books=None):
    s=requests.Session(); s.headers['User-Agent']='TurkishTextbookResearchBot/1.0'
    books=[]
    for grade in range(9,13):
        p=parse_html(fetch(s,f"{BASE}/{grade}-sinif-ders-ve-calisma-kitabi-cevaplari/"))
        for link in p.links:
            u=canonical(link); path=urlparse(u).path
            if re.match(fr"/{grade}-sinif-",path) and 'kitabi-cevaplari' in path and 'ders-ve-calisma' not in path:
                books.append((grade,u))
    books=list(dict.fromkeys(books)); return books[:max_books] if max_books else books

def discover_from_sitemaps(max_pages=None,workers=8):
    """Discover canonical 9-12 answer pages from every relevant category sitemap."""
    s=requests.Session();s.headers['User-Agent']='TurkishTextbookResearchBot/1.0'
    index=fetch(s,f"{BASE}/sitemap_index.xml")
    maps=[]
    for value in SITEMAP_LOC_RE.findall(index):
        url=html.unescape(value.strip());low=url.casefold()
        if 'kategori-kitap-cevaplari-' not in low:continue
        if not re.search(r'(?:^|-)\s*(?:9|10|11|12)-sinif(?:-|\.|$)',low):continue
        maps.append(url)
    maps=list(dict.fromkeys(maps));print(f'sitemaps={len(maps)}',flush=True)
    def one(url):
        session=requests.Session();session.headers['User-Agent']='TurkishTextbookResearchBot/1.0'
        try:return url,SITEMAP_LOC_RE.findall(fetch(session,url))
        except Exception as exc:
            print(f'sitemap_failed={url} {type(exc).__name__}:{exc}',flush=True);return url,[]
    found={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i,(source,values) in enumerate(ex.map(one,maps),1):
            for value in values:
                url=canonical(html.unescape(value.strip()));match=PAGE_RE.search(urlparse(url).path.strip('/'))
                if match:found[url]=(int(match.group('grade')),source,url)
            if i%25==0 or i==len(maps):print(f'sitemaps_done={i}/{len(maps)} pages={len(found)}',flush=True)
    items=list(found.values());items.sort(key=lambda x:(x[0],x[2]))
    return items[:max_pages] if max_pages else items

def page_links(book):
    grade,url=book; s=requests.Session(); s.headers['User-Agent']='TurkishTextbookResearchBot/1.0'
    p=parse_html(fetch(s,url)); out=[]
    for link in p.links:
        u=canonical(link); m=PAGE_RE.search(urlparse(u).path.strip('/'))
        if m and int(m.group('grade'))==grade: out.append(u)
    return grade,url,list(dict.fromkeys(out))

def scrape_page(item):
    grade,book_url,url=item; s=requests.Session(); s.headers['User-Agent']='TurkishTextbookResearchBot/1.0'
    body=fetch(s,url); p=parse_html(body); m=PAGE_RE.search(urlparse(url).path.strip('/'))
    title_match=re.search(r'<title>(.*?)</title>',body,re.I|re.S)
    content=p.article or p.main
    text='\n'.join(dict.fromkeys(html.unescape(x) for x in content if len(x)>1))
    # Remove the site's voting/comments tail; it is unrelated to textbook content.
    for marker in ('ile ilgili aşağıda bulunan emojileri', 'BU İÇERİĞE EMOJİYLE TEPKİ VER',
                   'Yorumlarınızı aşağıdaki bölümden yazabilirsiniz'):
        pos=text.find(marker)
        if pos >= 0: text=text[:pos]
    text=text.strip()
    empty_marker='Bu sayfada soru bulunmamaktadır'
    pos=text.casefold().find(empty_marker.casefold())
    if pos >= 0 and pos > 250:
        text=text[:pos].removesuffix('Cevap').strip()
    low=text.casefold()
    if empty_marker.casefold() in low:
        status='no_question'
    elif len(text) < 300 or ('bu soruyu çözmemizi istiyorsanız' in low and len(text) < 700):
        status='low_quality'
    else:
        status='useful'
    # Strip navigation/footer-heavy prefix by anchoring on the page title when possible.
    title=html.unescape(re.sub('<[^>]+>',' ',title_match.group(1))).strip() if title_match else ''
    image_urls=[]
    for raw in IMG_RE.findall(body):
        image_url=canonical(html.unescape(raw));path=urlparse(image_url).path.casefold()
        if f'sayfa-{m.group("page")}-' not in path:continue
        if not re.search(r'\.(?:jpe?g|png|webp)$',path):continue
        image_urls.append(image_url)
    image_urls=list(dict.fromkeys(image_urls))
    return {'source':'evvelcevap','grade':grade,'book_url':book_url,'page_url':url,
      'book_slug':m.group('book'),'page_number':int(m.group('page')),'title':title,
      'answer_text':text[:12000],'answer_status':status,'useful_answer':status=='useful',
      'image_urls':image_urls,'positive_image':None,
      'visual_status':'available' if image_urls else 'missing'}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--output',type=Path,required=True); a.add_argument('--max-books',type=int); a.add_argument('--max-pages',type=int); a.add_argument('--workers',type=int,default=4); a.add_argument('--resume',action='store_true'); a.add_argument('--discovery',choices=('categories','sitemaps'),default='categories'); args=a.parse_args()
    if args.discovery=='sitemaps':
        items=discover_from_sitemaps(args.max_pages,args.workers);books=[]
    else:
        books=discover(args.max_books); print(f"books={len(books)}",flush=True)
        book_pages=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i,res in enumerate(ex.map(page_links,books),1):
                book_pages.append(res); print(f"discover={i}/{len(books)} pages={sum(len(x[2]) for x in book_pages)}",flush=True)
        raw_items=[(g,u,p) for g,u,ps in book_pages for p in ps]
        items=list({p:(g,u,p) for g,u,p in raw_items}.values())
        if args.max_pages: items=items[:args.max_pages]
    done=set()
    if args.resume and args.output.exists():
        for line in args.output.open(encoding='utf-8'):
            try: done.add(json.loads(line)['page_url'])
            except Exception: pass
        items=[x for x in items if x[2] not in done]
        print(f"resume_done={len(done)} remaining={len(items)}",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True); good=failed=0
    with args.output.open('a' if args.resume else 'w',encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(scrape_page,x):x for x in items}
        for i,fut in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                row=fut.result(); f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); good+=1
            except Exception as e: failed+=1; print(f"failed={futures[fut][2]} {type(e).__name__}:{e}",flush=True)
            if i%25==0 or i==len(items): print(f"scraped={i}/{len(items)} good={good} failed={failed}",flush=True)
    print(json.dumps({'discovery':args.discovery,'books':len(books),'pages_discovered':len(items),'good':good,'failed':failed,'output':str(args.output)}),flush=True)
if __name__=='__main__': main()
