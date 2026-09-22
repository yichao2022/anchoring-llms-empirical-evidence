#!/usr/bin/env python3
"""Run 42 real block-task multinomial prompts through local Ollama qwen2.5:72b.

Each respondent sees 6 tasks within their block; tasks differ across blocks.
LLM is asked for P(A), P(B), P(C) (multinomial). 42 tasks x 10 reps = 420 calls.
Resume-capable (incremental CSV), temperature 0.7 per paper protocol.

Single source of task truth: dce_tasks_42_export.csv (columns A_*/B_*).
"""
import csv, json, re, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
TASKS_CSV = REPO / "dce_tasks_42_export.csv"
RAW_OUT = REPO / "llm_raw_outputs_multinomial_42block_tasks.csv"
PARSED_OUT = REPO / "llm_parsed_outputs_multinomial_42block_tasks.csv"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:72b"
TEMP = 0.7
NREP = 10
TIMEOUT = 300


def se_label(s):
    return {1: "0.1", 2: "1", 3: "10"}.get(int(float(s)), str(s))


def wait_label(w):
    w = float(w)
    return "walk-in (no wait)" if w == 0 else f"{int(w)} months"


def origin_label(o):
    return "Domestic" if int(o) == 0 else "Imported"


def profile_lines(p):
    return [
        f"- Waiting time: {wait_label(p['wait'])}",
        f"- Vaccine effectiveness: {int(round(float(p['eff'])*100))}%",
        f"- Risk of side effects: {se_label(p['se'])}%",
        f"- Cash incentive: {int(float(p['cash']))} RMB",
        f"- Vaccine origin: {origin_label(p['origin'])}",
    ]


def build_prompt(row):
    a = "\n".join(profile_lines(dict(wait=row['A_wait'], eff=row['A_eff_coded'], se=row['A_se_coded'], cash=row['A_cash'], origin=row['A_origin_coded'])))
    b = "\n".join(profile_lines(dict(wait=row['B_wait'], eff=row['B_eff_coded'], se=row['B_se_coded'], cash=row['B_cash'], origin=row['B_origin_coded'])))
    return (
        "You are evaluating COVID-19 vaccination choices.\n\n"
        "Please consider the following three alternatives and indicate which you would choose:\n\n"
        f"Alternative A:\n{a}\n\n"
        f"Alternative B:\n{b}\n\n"
        "Alternative C:\n- Do not receive the vaccine (opt out)\n\n"
        "Question: Which alternative would you choose?\n\n"
        "Provide your answer as probabilities that sum to 1:\n"
        "P(A): [probability]\nP(B): [probability]\nP(C): [probability]"
    )


def call_ollama(prompt):
    payload = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "options": {"temperature": TEMP}
    }).encode()
    req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())["message"]["content"]


def parse_response(text):
    pa = re.search(r"[*\s]*P\s*\(\s*A\s*\)[*\s]*[:=]?\s*([\d.]+)", text, re.I)
    pb = re.search(r"[*\s]*P\s*\(\s*B\s*\)[*\s]*[:=]?\s*([\d.]+)", text, re.I)
    pc = re.search(r"[*\s]*P\s*\(\s*C\s*\)[*\s]*[:=]?\s*([\d.]+)", text, re.I)
    choice = re.search(r"Choice:\s*([ABC])", text, re.I)
    def f(m):
        try:
            return float(m.group(1))
        except Exception:
            return None
    return {"P_A": f(pa), "P_B": f(pb), "P_C": f(pc),
            "choice": choice.group(1).upper() if choice else None}


def main():
    tasks = list(csv.DictReader(open(TASKS_CSV, encoding="utf-8")))
    print(f"tasks loaded: {len(tasks)}")
    raw_rows, parsed_rows = [], []
    done_keys = set()
    if RAW_OUT.exists():
        with open(RAW_OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                raw_rows.append(r)
                done_keys.add((int(r["task_num"]), int(r["repetition"])))
    if PARSED_OUT.exists():
        with open(PARSED_OUT, newline="", encoding="utf-8") as f:
            parsed_rows = list(csv.DictReader(f))
    print(f"resume: {len(done_keys)} done / {len(tasks)*NREP}")

    fw_raw = open(RAW_OUT, "a", newline="", encoding="utf-8") if raw_rows else open(RAW_OUT, "w", newline="", encoding="utf-8")
    fw_par = open(PARSED_OUT, "a", newline="", encoding="utf-8") if parsed_rows else open(PARSED_OUT, "w", newline="", encoding="utf-8")
    if not raw_rows:
        csv.writer(fw_raw).writerow(["task_num", "block_id", "repetition", "raw_response", "status"])
    if not parsed_rows:
        csv.writer(fw_par).writerow(["task_num", "block_id", "repetition", "P_A", "P_B", "P_C", "choice"])
    w_raw, w_par = csv.writer(fw_raw), csv.writer(fw_par)

    t0 = time.time()
    for row in tasks:
        tn = int(row["task_global_id"]); bid = int(row["block_id"])
        prompt = build_prompt(row)
        for rep in range(1, NREP + 1):
            if (tn, rep) in done_keys:
                continue
            try:
                text = call_ollama(prompt)
                status = "success"
            except Exception as e:
                text = f"ERROR: {e}"; status = "error"
            w_raw.writerow([tn, bid, rep, text, status]); fw_raw.flush()
            p = parse_response(text)
            w_par.writerow([tn, bid, rep, p["P_A"], p["P_B"], p["P_C"], p["choice"]]); fw_par.flush()
            done_keys.add((tn, rep))
            el = (time.time() - t0) / 60
            print(f"task {tn}/42 rep {rep}/10 done [{el:.1f}min]", flush=True)
    fw_raw.close(); fw_par.close()
    print("ALL DONE")


if __name__ == "__main__":
    main()
