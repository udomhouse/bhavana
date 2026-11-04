#!/usr/bin/env python3
"""Generate all chant and dhamma audio with Azure Speech, Thai voice "Nattapong" (MAI-Voice-2.1).

One voice for the whole app. Chants are paced like lay chanting: each phrase (วรรค — the words up to a
comma or the end of a line) is read in one breath, every phrase gets the same length per syllable, and
the voice pauses only between phrases and, longer, at the end of each line. Dhamma talks keep natural
speech. (The earlier Niwat version of this script is kept as generate_audio_azure_niwat.py.)

Newer Azure voices don't report word timings and decide their own pauses, so every phrase is synthesized
on its own (4 requests at a time, cached in scripts/.tts-cache so re-runs only fetch what changed). Inside
a phrase the voice's short breaths are tightened so the words flow; word highlight times are spread by
syllable count (accurate because the syllables are evenly paced). The slow style re-uses the same phrases
with longer pauses, a light stretch and room echo, so it is the very same voice.

Setup:
    pip install azure-cognitiveservices-speech numpy
    export AZURE_SPEECH_KEY=...            # Azure Portal › Speech resource › Keys and Endpoint › KEY 1
    export AZURE_SPEECH_REGION=southeastasia
Run:
    python3 scripts/generate_audio_azure.py                      # everything
    python3 scripts/generate_audio_azure.py --only itipiso,ariyasacca
Needs node (reads texts from www/app.js) and ffmpeg.
Output: www/audio/*.mp3 and www/audio/manifest.json (the format the app reads).
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from xml.sax.saxutils import escape
import numpy as np
import azure.cognitiveservices.speech as speechsdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'www', 'audio')
VOICE = 'th-TH-Nattapong:MAI-Voice-2.1'
SR = 24000
WORKERS = 4                # parallel requests
CACHE = os.path.join(ROOT, 'scripts', '.tts-cache')
TIGHTEN = (0.16, 0.07)     # inside a phrase, breaths longer than 0.16 s become 0.07 s

# Chant styles ("even"): syl_ms target length per syllable · comma_ms pause between phrases of a line ·
# gap pause at line end · stretch slows the slow style a little afterwards (pitch kept) · echo room echo.
# The voice's natural pace is close to 300 ms per syllable, so the normal style barely changes its speed.
STYLES = {
    'lay':   dict(even=True, syl_ms=280, comma_ms=700, gap=1000, echo=False, start_rate=1.0),
    'monk':  dict(even=True, syl_ms=280, comma_ms=650, gap=880,  echo=True,  start_rate=1.0, stretch=0.88),
    'topic': dict(even=False, rate=1.0, gap=900, echo=False),
}
PASSES = 5   # the voice reacts unevenly to speed changes: try gently, keep each phrase's closest take
VOWELS = set('ะาิีึืุูัเแโไใำ')

# How a word should sound when the written form misleads the voice (display text is unchanged).
PRONOUNCE = {
    'วิญญูหีติ': 'วินยูฮีติ',
    'วิญญูหิ': 'วินยูฮิ',
    'โหมิ': 'โห, มิ',         # two syllables โห-มิ; a plain space still came out as โห-หมิ
}
# Phrases to chant a little slower than the rest (factor on the syllable length).
SLOWER = {
    'อุชุปะฏิปันโน': 1.2,
    'โย วาจะโน ทินะทิเน': 1.2,
    'พุทธะมะอะอุ': 1.2,
    'ตะติยัมปิ ธัมมัง': 1.2,
}
TONE_MARKS = set('่้๊๋์')

def is_thai_prose(text):
    """Thai sentences (translations inside a chant) use tone marks; Pali written in Thai almost never does.
    Those lines are read at the voice's natural pace instead of the even chanting pulse."""
    return sum(c in TONE_MARKS for c in text) >= 2

def syllables(tok):
    """Pali written in Thai marks every syllable with a vowel sign (ภะ-คะ-วะ-โต), so counting them works."""
    return max(1, sum(c in VOWELS for c in tok))

def load_texts():
    js = subprocess.run(['node', os.path.join(ROOT, 'scripts', 'export-texts.mjs')], capture_output=True, text=True, check=True)
    return json.loads(js.stdout)

def spoken_text(t):
    t = t.replace('ฯ', '').replace('ฺ', '').replace('-', ' ')   # phinthu confuses the voice; ยะ-ธา-พุท is read per syllable
    t = re.sub(r'[;:()"“”]', ' ', t)
    for a, b in PRONOUNCE.items():
        t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t).strip()

def units_of(lines, even):
    """Spoken units in order: (line n, phrase p, [word indexes], [syllables per word], text).
    Chants: one unit per phrase. Talks: one unit per paragraph."""
    units, n = [], 0
    for line in lines:
        if line.startswith('~'):
            continue
        toks = [m.group() for m in re.finditer(r'\S+', line)]
        if even:
            p, cur = 0, []
            for wi, raw in enumerate(toks):
                cur.append(wi)
                if raw.endswith(',') or wi == len(toks) - 1:
                    words = [spoken_text(toks[i]).strip(',.') for i in cur]
                    text = ' '.join(w for w in words if w)
                    prose = is_thai_prose(text)
                    weights = [max(1, len(w) // 3) for w in words] if prose else [syllables(w) for w in words]
                    factor = max([f for k, f in SLOWER.items() if k in text] or [1.0])
                    units.append((n, p, cur, weights, text, prose, factor))
                    p, cur = p + 1, []
        else:
            units.append((n, 0, list(range(len(toks))), [syllables(spoken_text(t)) for t in toks], spoken_text(line), True, 1.0))
        n += 1
    return units, n

def decode(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def synthesize(cfg, ssml):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        path = tmp.name
    try:
        for attempt in range(6):
            synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=speechsdk.audio.AudioOutputConfig(filename=path))
            r = synth.speak_ssml_async(ssml).get()
            del synth
            if r.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return decode(path)
            det = r.cancellation_details
            msg = f'{det.reason} {det.error_details}' if det else str(r.reason)
            if '429' in msg or 'Too many' in msg or 'throttl' in msg.lower() or 'timeout' in msg.lower():
                time.sleep(15 * (attempt + 1)); continue
            raise RuntimeError(msg)
        raise RuntimeError('Azure kept throttling; try again later')
    finally:
        if os.path.exists(path):
            os.remove(path)

def silences(x, min_s=0.12):
    """[(start, end)] sample ranges of silence longer than min_s."""
    win = int(0.01 * SR)
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode='same'))
    thr = max(0.004, 0.06 * float(np.percentile(env, 98)))
    quiet = env < thr
    runs, i, n = [], 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_s * SR:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs

def split_units(x, count):
    """Cut audio holding `count` units separated by long pauses into `count` speech spans."""
    runs = silences(x)
    inner = [r for r in runs if r[0] > 0 and r[1] < len(x)]
    if len(inner) < count - 1:
        raise RuntimeError(f'found {len(inner)} pauses, need {count - 1}')
    by_len = sorted(inner, key=lambda r: r[1] - r[0], reverse=True)
    cut = sorted(by_len[:count - 1])
    if count > 1 and len(by_len) >= count:
        chosen = by_len[count - 2][1] - by_len[count - 2][0]
        other = by_len[count - 1][1] - by_len[count - 1][0]
        if chosen < other * 1.25:
            raise RuntimeError(f'phrase pauses unclear ({chosen / SR:.2f}s vs {other / SR:.2f}s)')
    lead = runs[0][1] if runs and runs[0][0] == 0 else 0
    tail = runs[-1][0] if runs and runs[-1][1] >= len(x) else len(x)
    edges = [lead] + [v for a, b in cut for v in (a, b)] + [tail]
    return [(edges[2 * i], edges[2 * i + 1]) for i in range(count)]

def tighten(seg):
    """Shorten the voice's own breaths inside one phrase so its words flow together."""
    out, last = [], 0
    keep = int(TIGHTEN[1] * SR)
    for a, b in silences(seg, TIGHTEN[0]):
        if a == 0 or b >= len(seg):
            continue
        out.append(seg[last:a + keep // 2]); last = b - keep // 2
    out.append(seg[last:])
    return np.concatenate(out)

def stretch_seg(seg, tempo):
    """Slow one phrase down (tempo < 1), pitch kept."""
    out = subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                          '-af', f'atempo={tempo:.3f}', '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         input=seg.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()

def pad(seg, pre=0.03, post=0.06):
    return np.concatenate([np.zeros(int(pre * SR), np.float32), seg, np.zeros(int(post * SR), np.float32)])

def trim(x):
    """Cut leading and trailing silence."""
    runs = silences(x, 0.02)
    a = runs[0][1] if runs and runs[0][0] == 0 else 0
    b = runs[-1][0] if runs and runs[-1][1] >= len(x) else len(x)
    return x[a:b] if b > a else x

def speak(cfg, text, rate):
    """One phrase → audio (cached by voice, rate and text)."""
    key = hashlib.sha1(f'{VOICE}|{rate:.3f}|{text}'.encode()).hexdigest()
    path = os.path.join(CACHE, key + '.f32')
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.float32)
    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
            f'<voice name="{VOICE}"><prosody rate="{rate:.3f}">{escape(text)}</prosody></voice></speak>')
    x = trim(synthesize(cfg, ssml))
    os.makedirs(CACHE, exist_ok=True)
    x.astype(np.float32).tofile(path)
    return x

def render_units(cfg, units, st, rates):
    """Synthesize every unit on its own and return one audio array per unit."""
    def one(i):
        u = units[i]
        base = st.get('start_rate', st.get('rate', 1.0))
        r = round(base if u[5] else rates.get(i, base / u[6]), 2)
        x = speak(cfg, u[4], r)
        return tighten(x) if st['even'] and not u[5] else x
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(one, range(len(units))))

def encode(y, path, st):
    filters = []
    if st.get('stretch', 1.0) != 1.0:
        filters.append(f"atempo={st['stretch']}")
    if st['echo']:
        filters.append('aecho=0.8:0.55:70|140:0.22|0.12')
    filters.append('loudnorm=I=-18:TP=-2:LRA=7')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                    '-af', ','.join(filters), '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '40k', path],
                   input=y.astype(np.float32).tobytes(), check=True)

def assemble(path, units, segs, nlines, st):
    """Lay the units back to back with fixed pauses; return (cues, word anchors)."""
    sil = lambda ms: np.zeros(int(SR * ms / 1000), dtype=np.float32)
    fade = int(0.01 * SR)
    out, t = [sil(300)], 0.3
    cues, anchors = [None] * nlines, [[] for _ in range(nlines)]
    for i, (u, seg) in enumerate(zip(units, segs)):
        n, p, wis, syl = u[:4]
        seg = pad(seg).copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        if cues[n] is None:
            cues[n] = round(t, 2)
        if st['even']:   # spread word starts by syllables across the spoken part
            speak = len(seg) / SR - 0.09
            total, acc = sum(syl), 0
            for wi, sy in zip(wis, syl):
                anchors[n].append([wi, round(t + 0.03 + speak * acc / total, 2)])
                acc += sy
        out.append(seg); t += len(seg) / SR
        last_of_line = i == len(units) - 1 or units[i + 1][0] != n
        ms = st['gap'] if last_of_line else st['comma_ms']
        out.append(sil(ms)); t += ms / 1000
    encode(np.concatenate(out), path, st)
    k = 1 / st.get('stretch', 1.0)
    return [round((c or 0) * k, 2) for c in cues], [[[wi, round(tt * k, 2)] for wi, tt in la] for la in anchors]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma-separated ids to regenerate')
    args = ap.parse_args()
    key, region = os.environ.get('AZURE_SPEECH_KEY'), os.environ.get('AZURE_SPEECH_REGION')
    if not key or not region:
        sys.exit('Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION first.')
    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if proxy:  # the Speech SDK does not read proxy variables on its own
        from urllib.parse import urlparse
        u = urlparse(proxy)
        cfg.set_proxy(u.hostname, u.port or 80, u.username or '', u.password or '')
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3)

    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, 'manifest.json')
    manifest = json.load(open(man_path, encoding='utf8')) if os.path.exists(man_path) else {}
    only = set(filter(None, args.only.split(',')))
    data = load_texts()
    jobs = [(c['id'], c['lines'], ['lay', 'monk']) for c in data['chants']] + \
           [(t['id'], t['lines'], ['topic']) for t in data['topics']]
    failed = []
    for key_id, lines, styles in jobs:
        if only and key_id not in only and key_id.replace('topic:', '') not in only:
            continue
        entry = {'cues': {}, 'words': {}}
        lay_segs = None
        try:
            for sname in styles:
                st = STYLES[sname]
                slot = 'lay' if sname == 'topic' else sname
                fname = (key_id.replace(':', '-') + '.mp3') if sname == 'topic' else f'{key_id}-{sname}.mp3'
                units, nlines = units_of(lines, st['even'])
                rates, note = {}, ''
                if sname == 'monk' and lay_segs is not None:   # same voice, same phrases: only pauses/stretch/echo differ
                    segs, note = lay_segs, ' (from normal style)'
                    cues, anchors = assemble(os.path.join(OUT, fname), units, segs, nlines, st)
                    note += f' → stretched to ~{st["syl_ms"] / st["stretch"]:.0f}ms'
                    entry[slot] = fname; entry['cues'][slot] = cues; entry['words'][slot] = anchors
                    print(f'{fname:30s} {nlines} lines · {len(units)} units{note}', flush=True)
                    continue
                best = {}
                for p in range(PASSES if st['even'] else 1):
                    segs = render_units(cfg, units, st, rates)
                    if not st['even']:
                        break
                    want = [sum(u[3]) * st['syl_ms'] / 1000 * u[6] for u in units]
                    for i, (sg, w) in enumerate(zip(segs, want)):
                        err = 0 if units[i][5] else abs(len(sg) / SR / w - 1)   # Thai prose: natural pace, keep as is
                        if i not in best or err < best[i][0]:
                            best[i] = (err, sg)
                    chant = [i for i, u in enumerate(units) if not u[5]]
                    per = [(len(best[i][1]) / SR) / sum(units[i][3]) / units[i][6] for i in chant] or [1]
                    mean = sum(per) / len(per)
                    cv = (sum((v - mean) ** 2 for v in per) / len(per)) ** 0.5 / mean * 100
                    note += f' pass{p + 1}: {mean * 1000:.0f}ms±{cv:.0f}%'
                    for i, (sg, w) in enumerate(zip(segs, want)):   # damped correction
                        if units[i][5]:
                            continue
                        cur = rates.get(i, st['start_rate'] / units[i][6])
                        floor = 0.4 if units[i][6] > 1 else 0.6   # phrases asked to be slower may go further
                        rates[i] = min(1.6, max(floor, cur * (len(sg) / SR / w) ** 0.7))
                if st['even']:
                    segs = [best[i][1] for i in range(len(units))]
                    for i, u in enumerate(units):   # a phrase asked to be slower that the voice still rushes: stretch only it
                        w = sum(u[3]) * st['syl_ms'] / 1000 * u[6]
                        if u[6] > 1 and not u[5] and len(segs[i]) / SR < w * 0.93:
                            segs[i] = stretch_seg(segs[i], max(0.75, len(segs[i]) / SR / w))
                if sname == 'lay':
                    lay_segs = segs
                cues, anchors = assemble(os.path.join(OUT, fname), units, segs, nlines, st)
                if st.get('stretch', 1.0) != 1.0:
                    note += f' → stretched to ~{st["syl_ms"] / st["stretch"]:.0f}ms'
                entry[slot] = fname
                entry['cues'][slot] = cues
                if st['even']:
                    entry['words'][slot] = anchors
                print(f'{fname:30s} {nlines} lines · {len(units)} units{note}', flush=True)
        except Exception as e:
            print(f'FAILED {key_id}: {e}', flush=True); failed.append(key_id); continue
        if not entry['words']:
            del entry['words']
        manifest[key_id] = entry
    manifest['_readme'] = f'สร้างโดย scripts/generate_audio_azure.py · เสียง Azure "{VOICE}" · ใส่ไฟล์เสียงอัดจริงแทนได้ ดู README.md'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print('manifest:', man_path, '| failed:', failed or 'none')

if __name__ == '__main__':
    main()
