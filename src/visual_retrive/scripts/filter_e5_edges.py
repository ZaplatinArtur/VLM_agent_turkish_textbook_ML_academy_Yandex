"""Filter a mined E5 edge file to a selected semantic/lexical threshold."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--semantic-threshold',type=float,required=True); ap.add_argument('--lexical-threshold',type=float,default=.72)
a=ap.parse_args(); data=json.loads(a.input.read_text(encoding='utf-8'))
data['edges']=[e for e in data['edges'] if float(e.get('semantic') or 0)>=a.semantic_threshold or float(e.get('lexical') or 0)>=a.lexical_threshold]
data['config']={**data['config'],'semantic_threshold':a.semantic_threshold,'lexical_threshold':a.lexical_threshold}
data['stats']={**data['stats'],'edges':len(data['edges'])}; a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data['stats']))
