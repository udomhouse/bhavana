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
VOWELS = set('ะาิีึืุูัเแโไใำ')

# How a word should sound when the written form misleads the voice (display text is unchanged).
PRONOUNCE = {
    'วิญญูหีติ': 'วินยูฮีติ',
    'วิญญูหิ': 'วินยูฮิ',
    'โหมิ': 'โห, มิ',         # two syllables โห-มิ; a plain space still came out as โห-หมิ
    # listener's corrections, Nov 2026
    'สัจจะวัชเชนะ': 'สัดจะ วัดเชนะ',
    'เอกัสมิง': 'เอกัดสะ มิง',
    'ปะหีนา': 'ปะฮีนา',
    'บูระพารัสมิง': 'บูระพา รัดสะหมิง',
    'อาคะเนยรัสมิง': 'อาคะเน รัดสะหมิง',
    'หรดีรัสมิง': 'หอระดี รัดสะหมิง',
    'อุดรรัสมิง': 'อุดอน รัดสะหมิง',
    'อากาศรัสมิง': 'อากาด รัดสะหมิง',
    'รัสมิง': 'รัดสะหมิง',      # the other directions
    'ธาตุ': 'ทา ตุ',
    'หุลู': 'หุ, ลู',          # keep ลู level (not หลู)
    'เวทะนาขันโธ': 'เวดทะนา ขันโธ',
    'ฏฐ': 'ตถ',               # ปะติฏฐาสิ → ปะ-ติด-ถา-สิ (must come before ฏ)
    'ฏ': 'ต',                  # สุปะฏิปันโน → สุ-ปะ-ติ-ปัน-โน
    'โหตุ': 'โห ตุ',
    'มารดา': 'มานดา',
    'โสตถ': 'โสดถ',            # โสตถี / โสตถิ
    'วินาศสันติ': 'วินาดสันติ',
    'พรหมา': 'พรัมมา',
    'พรหมะ': 'พรัมมะ',
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
    t = re.sub(r'(\S+?)ๆ', r'\1 \1', t)   # ไม้ยมก: เพ็งๆ → เพ็ง เพ็ง
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
    xf = int(0.012 * SR)   # soft splice: tiny crossfade so cut breaths never click or ring
    y = out[0]
    for part in out[1:]:
        if len(y) > xf and len(part) > xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            y = np.concatenate([y[:-xf], y[-xf:] * (1 - ramp) + part[:xf] * ramp, part[xf:]])
        else:
            y = np.concatenate([y, part])
    return y

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

def tone_glitch(x):
    """Seconds of the longest pure-tone stretch: the voice model sometimes hums a steady sine
    (heard as ringing or a bell) instead of speech. Speech always has harmonics; a tone does not."""
    n, h = int(0.04 * SR), int(0.02 * SR)
    win = np.hanning(n).astype(np.float32)
    longest = run = 0
    floor = 0.12 * float(np.sqrt((x ** 2).mean())) if len(x) else 0   # relative: takes are not normalised yet
    for s in range(0, len(x) - n, h):
        fr = x[s:s + n]
        if np.sqrt((fr ** 2).mean()) < floor:
            run = 0; continue
        sp = np.abs(np.fft.rfft(fr * win)) ** 2
        k = int(np.argmax(sp[3:])) + 3
        run = run + 1 if sp[max(0, k - 2):k + 3].sum() / sp.sum() > 0.85 else 0
        longest = max(longest, run)
    return longest * h / SR

GLITCH_S = 0.12
JITTER = [0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04]
glitch_log = []

def speak_clean(cfg, text, rate):
    """speak(), retried at a hair's different speed while the take contains a pure-tone glitch."""
    best = None
    for d in JITTER:
        x = speak(cfg, text, round(rate + d, 2))
        g = tone_glitch(x)
        if best is None or g < best[0]:
            best = (g, x)
        if g < GLITCH_S:
            if d: glitch_log.append(f'fixed {text[:24]}')
            return x
    glitch_log.append(f'STILL {best[0]:.2f}s {text[:24]}')
    return best[1]

def render_units(cfg, units, st, rates):
    """Synthesize every unit on its own and return one audio array per unit."""
    def one(i):
        u = units[i]
        base = st.get('start_rate', st.get('rate', 1.0))
        r = round(base if u[5] else rates.get(i, base / u[6]), 2)
        x = speak_clean(cfg, u[4], r)
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
                    '-af', ','.join(filters), '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', path],
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
            tail = min(int(0.035 * SR), len(seg) // 4)   # let the last word die away, no abrupt cut
            seg[-tail:] *= np.linspace(1, 0, tail, dtype=np.float32) ** 2
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
    # the slow 'monk' style was removed from the app
    jobs = [(c['id'], c['lines'], ['lay']) for c in data['chants']] + \
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
                want = [sum(u[3]) * st['syl_ms'] / 1000 * u[6] for u in units]
                chant = [i for i, u in enumerate(units) if not u[5]]
                def score(segs):
                    for i, sg in enumerate(segs):
                        err = 0 if units[i][5] else abs(len(sg) / SR / want[i] - 1)   # Thai prose: natural pace
                        if i not in best or err < best[i][0]:
                            best[i] = (err, sg)
                    per = [(len(best[i][1]) / SR) / sum(units[i][3]) / units[i][6] for i in chant] or [1]
                    mean = sum(per) / len(per)
                    return f'{mean * 1000:.0f}ms±{(sum((v - mean) ** 2 for v in per) / len(per)) ** 0.5 / mean * 100:.0f}%'
                if not st['even']:
                    segs = render_units(cfg, units, st, rates)
                else:
                    # 1) probe at normal speed, 2) ONE speed for the whole chant (big speed swings change
                    # the voice's timbre and sound like another speaker), 3) small ±8% touch-ups per phrase.
                    probe = render_units(cfg, units, st, {i: 1.0 / units[i][6] for i in chant})
                    ratio = sorted(len(probe[i]) / SR / want[i] for i in chant) or [1]
                    g = min(1.15, max(0.85, ratio[len(ratio) // 2]))
                    rates = {i: g / units[i][6] for i in chant}
                    segs = render_units(cfg, units, st, rates)
                    note += f' global rate {g:.2f}: {score(segs)}'
                    fix = {}
                    for i in chant:
                        d = len(segs[i]) / SR / want[i]
                        if abs(d - 1) > 0.04:
                            fix[i] = rates[i] * min(1.08, max(0.92, d))
                    if fix:
                        rates.update(fix)
                        segs2 = render_units(cfg, units, st, rates)
                        note += f' → touch-up {len(fix)}: {score(segs2)}'
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
                if glitch_log:
                    print('   glitches:', ' | '.join(dict.fromkeys(glitch_log)), flush=True); glitch_log.clear()
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
