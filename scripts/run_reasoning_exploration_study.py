#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from cot_research.answer_extraction import answers_match, classify_outcome, extract_last_boxed

REFLECTION_PATTERNS = [
    ("wait", re.compile(r"(?i)(?<![a-z])wait(?![a-z])")),
    ("hold_on", re.compile(r"(?i)hold on|one moment|hang on")),
    ("actually", re.compile(r"(?i)(?<![a-z])actually(?![a-z])")),
    ("however", re.compile(r"(?i)(?<![a-z])however(?![a-z])")),
    ("reconsider", re.compile(r"(?i)reconsider|on second thought|let me reconsider")),
    ("check", re.compile(r"(?i)let me check|check again|recheck|verify|double-check")),
    ("mistake", re.compile(r"(?i)mistake|incorrect|wrong|error|not correct")),
    ("cn_reflect", re.compile(r"不对|等等|等一下|重新|检查|错误|有误|矛盾")),
]
TRANSITION_PATTERNS = [
    ("therefore", re.compile(r"(?i)\btherefore\b|\bthus\b|\bhence\b")),
    ("so_we_have", re.compile(r"(?i)\bso we have\b|\bso,\s*we\b")),
    ("now", re.compile(r"(?i)\bnow\b|\bnext\b")),
    ("finally", re.compile(r"(?i)\bfinally\b|\bto conclude\b")),
]
STRATEGY_PATTERNS = [
    ("geometry", re.compile(r"(?i)triangle|circle|angle|radius|area|perimeter|similar|coordinate|slope")),
    ("algebra", re.compile(r"(?i)equation|solve for|substitute|expand|factor|quadratic|linear|system")),
    ("number_theory", re.compile(r"(?i)modulo|divisible|prime|gcd|lcm|remainder|integer")),
    ("casework", re.compile(r"(?i)case 1|case 2|cases|split into")),
    ("counting", re.compile(r"(?i)probability|choose|permutation|combination|count")),
    ("calculation", re.compile(r"(?i)compute|calculate|simplify|sum|product")),
]


def load_jsonl(path):
    rows=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def dump_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')

def append_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')

def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k.endswith('completion_text') or k == 'prompt_text': continue
            if k not in seen: seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})

def row_id(row, fallback):
    for k in ['id','example_id','unique_id','problem_id','idx']:
        if row.get(k) is not None: return str(row[k])
    return str(fallback)

def row_question(row):
    for k in ['question','problem','prompt']:
        if row.get(k): return str(row[k]).strip()
    return ''

def row_answer(row):
    for k in ['correct_answer','answer','target']:
        if row.get(k) is not None: return str(row[k]).strip()
    return None

def think_sections(text):
    lo=text.lower(); s=lo.find('<think>'); e=lo.find('</think>')
    if s>=0 and e>=s: return text[s+7:e], text[e+8:]
    if e>=0: return text[:e], text[e+8:]
    return text, ''

def find_hits(patterns, text):
    hits=[]
    for kind,pat in patterns:
        for m in pat.finditer(text): hits.append({'kind':kind,'start':m.start(),'end':m.end(),'text':m.group(0)})
    return sorted(hits, key=lambda x:(x['start'],x['end']))

def boxed_spans(text):
    out=[]
    for m in re.finditer(r"\\boxed\{", text):
        depth=1; p=m.end()
        while p < len(text) and depth>0:
            if text[p]=='{': depth+=1
            elif text[p]=='}': depth-=1
            p+=1
        if depth==0: out.append((m.start(), p, text[m.end():p-1]))
    return out

def answer_candidates(text):
    out=[{'kind':'boxed','start':s,'end':e,'value':v} for s,e,v in boxed_spans(text)]
    pat=re.compile(r"(?i)(?:answer is|final answer is|the result is|we get|equals|=)\s*[:$ ]*(-?\\?frac\{[^}]+\}\{[^}]+\}|-?\d+(?:\.\d+)?(?:/\d+)?|[A-E])")
    for m in pat.finditer(text): out.append({'kind':'phrase','start':m.start(1),'end':m.end(1),'value':m.group(1)})
    return sorted(out, key=lambda x:(x['start'],x['end']))

def strategy_label(text):
    c=Counter({name: len(pat.findall(text)) for name,pat in STRATEGY_PATTERNS})
    return 'unknown' if not c or c.most_common(1)[0][1]==0 else c.most_common(1)[0][0]

def highlight_reflections(text, context=None):
    hits=find_hits(REFLECTION_PATTERNS, text)
    if context and hits:
        s=max(0,hits[0]['start']-context); e=min(len(text), hits[-1]['end']+context)
        text=('[...]\n' if s else '')+text[s:e]+('\n[...]' if e<len(text) else '')
        hits=find_hits(REFLECTION_PATTERNS, text)
    parts=[]; last=0
    for h in hits:
        parts.append(text[last:h['start']]); parts.append(f">>>REFLECT[{h['kind']}]: {text[h['start']:h['end']]}<<<"); last=h['end']
    parts.append(text[last:]); return ''.join(parts)

def analyze_text(text, correct, token_ids=None):
    think, final=think_sections(text); refs=find_hits(REFLECTION_PATTERNS, think); trans=find_hits(TRANSITION_PATTERNS, think); cands=answer_candidates(think)
    final_box=extract_last_boxed(text); final_correct=bool(final_box and correct and answers_match(final_box, correct))
    first=refs[0]['start'] if refs else None
    pre=[c for c in cands if first is not None and c['end']<=first]; post=[c for c in cands if first is not None and c['start']>=first]
    pre_v=pre[-1]['value'] if pre else None; post_v=post[-1]['value'] if post else final_box
    pre_ok=bool(pre_v and correct and answers_match(pre_v, correct)); post_ok=bool(post_v and correct and answers_match(post_v, correct))
    changed=bool(pre_v and post_v and not answers_match(pre_v, post_v))
    if first is None: rt='no_reflection'
    elif pre_v is None: rt='reflection_no_prior_answer'
    elif (not pre_ok) and final_correct: rt='wrong_to_right'
    elif pre_ok and not final_correct: rt='right_to_wrong'
    elif changed: rt='answer_changed_other'
    else: rt='no_answer_change'
    return {'generated_tokens':len(token_ids or []),'generated_chars':len(text),'think_chars':len(think),'final_chars':len(final),'has_think_end':'</think>' in text.lower(),'has_reflection':bool(refs),'reflection_count':len(refs),'first_reflection_kind':refs[0]['kind'] if refs else None,'first_reflection_pos':first,'matched_reflection_kinds':sorted({h['kind'] for h in refs}),'transition_count':len(trans),'first_transition_kind':trans[0]['kind'] if trans else None,'pre_reflection_answer':pre_v,'post_reflection_answer':post_v,'pre_reflection_answer_correct':pre_ok if pre_v is not None else None,'post_reflection_answer_correct':post_ok if post_v is not None else None,'answer_changed_after_reflection':changed if first is not None else None,'reflection_transition':rt,'final_boxed_answer':final_box,'final_correct':final_correct,'outcome':classify_outcome(final_box, correct, None),'strategy_label':strategy_label(think),'step_count_proxy':max(len([x for x in think.splitlines() if x.strip()]), len(trans))}

def summarize(rows):
    def mean(vals):
        xs=[]
        for v in vals:
            if v is None: continue
            try: x=float(v)
            except Exception: continue
            if math.isfinite(x): xs.append(x)
        return sum(xs)/len(xs) if xs else None
    return {'count':len(rows),'final_correct_rate':mean([r.get('final_correct') for r in rows]),'has_reflection_rate':mean([r.get('has_reflection') for r in rows]),'mean_reflection_count':mean([r.get('reflection_count') for r in rows]),'mean_generated_tokens':mean([r.get('generated_tokens') for r in rows]),'reflection_transition_counts':dict(Counter(str(r.get('reflection_transition')) for r in rows)),'strategy_counts':dict(Counter(str(r.get('strategy_label')) for r in rows))}

def build_prompt(tokenizer, question, system_prompt, enable_thinking=True):
    msgs=[{'role':'system','content':system_prompt},{'role':'user','content':question}]
    try: return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError: return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def restatement_variants(q):
    return [('original',q),('plain','Solve this carefully in a direct way. Problem: '+q),('distractor','Solve the problem below. Extra note: some numbers in contest problems may be irrelevant, so verify which information matters. Problem: '+q),('equivalent_frame','Find the mathematical quantity requested by the following equivalent formulation, and give the final result in a box: '+q)]
