#!/usr/bin/env python3
"""Generate all chant and dhamma audio with Azure Speech, Thai voice "Niwat" (th-TH-NiwatNeural).

One voice for the whole app. Chants are paced like lay chanting: each phrase (วรรค — the words up to a
comma or the end of a line) is read in one breath, every phrase gets the same speed per syllable, and
the voice pauses only between phrases and, longer, at the end of each line. Dhamma talks keep natural
speech. Azure reports when every line and word is spoken, so the app highlights the word being chanted.

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
import argparse, json, os, re, subprocess, sys, time
from xml.sax.saxutils import escape
import numpy as np
import azure.cognitiveservices.speech as speechsdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'www', 'audio')
VOICE = 'th-TH-NiwatNeural'
SR = 24000

# Chant styles ("even"):
#   syl_ms   target length of one syllable inside a phrase
#   comma_ms pause between phrases of the same line · gap: pause at the end of a line
#   stretch  Azure can't go below half speed, so the slow style is time-stretched afterwards (pitch kept)
STYLES = {
    # The voice is left as Azure speaks it: no pitch shift, and the normal style is not time-stretched —
    # both bent the Thai tones and made Niwat sound foreign. Read continuously, Azure's slowest even pace
    # is ~300 ms per syllable; calm comes from long breaths between phrases and lines instead.
    # The slow style keeps a light stretch (pitch kept) plus room echo.
    'lay':   dict(even=True, syl_ms=300, comma_ms=700, gap=1500, pitch='+0%', echo=False, start_rate=0.65),
    'monk':  dict(even=True, syl_ms=300, comma_ms=650, gap=1350, pitch='+0%', echo=True,  start_rate=0.65, stretch=0.88),
    'topic': dict(even=False, rate='-8%', pitch='+0%', gap=700, echo=False),
}
PASSES = 3   # 1 measuring pass + 2 corrections
VOWELS = set('ะาิีึืุูัเแโไใำ')

def syllables(tok):
    """Pali written in Thai marks every syllable with a vowel sign (ภะ-คะ-วะ-โต), so counting them works."""
    return max(1, sum(c in VOWELS for c in tok))

def load_texts():
    js = subprocess.run(['node', os.path.join(ROOT, 'scripts', 'export-texts.mjs')], capture_output=True, text=True, check=True)
    return json.loads(js.stdout)

def spoken_text(line):
    t = line.replace('ฯ', '').replace('ฺ', '')   # phinthu (เทฺวเม) confuses the voice; drop it
    t = re.sub(r'[;:()"“”]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def phrases_of(line):
    """[[(word index, token)], ...] — a phrase ends at a token with a comma, or at the end of the line."""
    out, cur = [], []
    toks = list(re.finditer(r'\S+', line))
    for wi, m in enumerate(toks):
        raw = m.group()
        tok = spoken_text(raw).strip(',.')
        if tok:
            cur.append((wi, tok))
        if raw.endswith(',') and cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out

def build_ssml(lines, st, rates=None):
    """Returns (ssml, words, phrases).
    words[n]   = [(start, end, word index, syllables)] character ranges of each word in the ssml
    phrases[n] = [[word index, ...], ...] for even styles"""
    lead = '<break time="300ms"/>'
    if st['even']:
        head = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
                f'<voice name="{VOICE}"><prosody pitch="{st["pitch"]}">{lead}')
    else:
        head = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
                f'<voice name="{VOICE}"><prosody rate="{st["rate"]}" pitch="{st["pitch"]}">{lead}')
    ssml, words, phrases, n = head, [], [], 0
    for line in lines:
        if line.startswith('~'):
            continue
        ssml += f'<bookmark mark="L{n}"/>'
        offs, line_ph = [], []
        if st['even']:
            ph = phrases_of(line)
            for p, group in enumerate(ph):
                r = (rates or {}).get((n, p), st['start_rate'])
                ssml += f'<prosody rate="{r:.3f}">'
                for j, (wi, tok) in enumerate(group):
                    if j:
                        ssml += ' '      # inside one prosody element a space does not make Azure pause
                    a = len(ssml); ssml += escape(tok)
                    offs.append((a, len(ssml), wi, syllables(tok)))
                ssml += '</prosody>'
                if p < len(ph) - 1:
                    ssml += f'<break time="{st["comma_ms"]}ms"/>'
                line_ph.append([wi for wi, _ in group])
        else:
            for wi, m in enumerate(re.finditer(r'\S+', line)):
                tok = spoken_text(m.group())
                if not tok:
                    continue
                if offs:
                    ssml += ' '
                a = len(ssml); ssml += escape(tok)
                offs.append((a, len(ssml), wi, syllables(tok)))
        ssml += f'<break time="{st["gap"]}ms"/>'
        words.append(offs); phrases.append(line_ph)
        n += 1
    ssml += '</prosody></voice></speak>'
    return ssml, words, phrases

def measure(words, bounds):
    """Per word: (start time, end time) from the Azure word boundaries that fall inside it."""
    out = {}
    for n, offs in enumerate(words):
        for a, b, wi, _ in offs:
            seg = [(t, t + d) for off, t, d in bounds if a <= off < b]
            if seg:
                out[(n, wi)] = (min(x for x, _ in seg), max(y for _, y in seg))
    return out

def phrase_times(words, phrases, timing):
    """Per phrase: (start, end, syllables), or None when no word in it was timed."""
    syl = {(n, wi): sy for n, offs in enumerate(words) for _, _, wi, sy in offs}
    res = {}
    for n, line_ph in enumerate(phrases):
        for p, wis in enumerate(line_ph):
            ts = [timing[(n, wi)] for wi in wis if (n, wi) in timing]
            res[(n, p)] = (min(a for a, _ in ts), max(b for _, b in ts), sum(syl[(n, wi)] for wi in wis)) if ts else None
    return res

def synthesize(cfg, ssml, out_mp3):
    audio_cfg = speechsdk.audio.AudioOutputConfig(filename=out_mp3)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
    marks, bounds = {}, []
    synth.bookmark_reached.connect(lambda e: marks.__setitem__(e.text, e.audio_offset / 1e7))
    synth.synthesis_word_boundary.connect(lambda e: bounds.append((e.text_offset, e.audio_offset / 1e7, e.duration.total_seconds())))
    for attempt in range(5):
        r = synth.speak_ssml_async(ssml).get()
        if r.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            del synth
            return marks, bounds
        det = r.cancellation_details
        msg = f'{det.reason} {det.error_details}' if det else str(r.reason)
        if '429' in msg or 'Too many' in msg or 'throttl' in msg.lower():
            time.sleep(15 * (attempt + 1)); marks.clear(); bounds.clear(); continue
        raise RuntimeError(msg)
    raise RuntimeError('Azure kept throttling; try again later')

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

def decode(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def rebuild(path, words, phrases, timing, st):
    """Cut every phrase out of Azure's audio and lay the phrases back to back with fixed pauses
    (comma_ms between phrases, gap at line end), so the breathing rhythm is the same everywhere.
    Returns (cues, anchors) in the rebuilt, possibly stretched, audio."""
    x = decode(path)
    sil = lambda ms: np.zeros(int(SR * ms / 1000), dtype=np.float32)
    fade = int(0.01 * SR)
    pt = phrase_times(words, phrases, timing)
    out, t, cues, anchors = [sil(300)], 0.3, [], []
    for n, line_ph in enumerate(phrases):
        cues.append(round(t, 2))
        line_a = []
        for p, wis in enumerate(line_ph):
            if pt[(n, p)] is None:
                continue
            s0, e0, _ = pt[(n, p)]
            a = max(0, int((s0 - 0.03) * SR)); b = min(len(x), int((e0 + 0.08) * SR))
            seg = x[a:b].copy()
            if len(seg) > 2 * fade:
                seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            for wi in wis:
                if (n, wi) in timing:
                    line_a.append([wi, round(t + timing[(n, wi)][0] - a / SR, 2)])
            out.append(seg); t += len(seg) / SR
            ms = st['gap'] if p == len(line_ph) - 1 else st['comma_ms']
            out.append(sil(ms)); t += ms / 1000
        anchors.append(line_a)
    encode(np.concatenate(out), path, st)
    k = 1 / st.get('stretch', 1.0)
    return [round(c * k, 2) for c in cues], [[[wi, round(tt * k, 2)] for wi, tt in la] for la in anchors]

def add_echo(mp3):
    encode(decode(mp3), mp3, {'echo': True})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma-separated ids to regenerate')
    args = ap.parse_args()
    key, region = os.environ.get('AZURE_SPEECH_KEY'), os.environ.get('AZURE_SPEECH_REGION')
    if not key or not region:
        sys.exit('Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION first.')
    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.speech_synthesis_voice_name = VOICE
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if proxy:  # the Speech SDK does not read proxy variables on its own
        from urllib.parse import urlparse
        u = urlparse(proxy)
        cfg.set_proxy(u.hostname, u.port or 80, u.username or '', u.password or '')
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3)

    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, 'manifest.json')
    manifest = json.load(open(man_path, encoding='utf8')) if os.path.exists(man_path) else {}
    only = set(filter(None, args.only.split(',')))
    data = load_texts()
    jobs = [(c['id'], c['lines'], ['lay', 'monk']) for c in data['chants']] + \
           [(t['id'], t['lines'], ['topic']) for t in data['topics']]
    for key_id, lines, styles in jobs:
        if only and key_id not in only and key_id.replace('topic:', '') not in only:
            continue
        entry = {'cues': {}, 'words': {}}
        for sname in styles:
            st = STYLES[sname]
            slot = 'lay' if sname == 'topic' else sname
            fname = (key_id.replace(':', '-') + '.mp3') if sname == 'topic' else f'{key_id}-{sname}.mp3'
            path = os.path.join(OUT, fname)
            rates, note = {}, ''
            for p in range(PASSES if st['even'] else 1):
                ssml, words, phrases = build_ssml(lines, st, rates)
                marks, bounds = synthesize(cfg, ssml, path)
                timing = measure(words, bounds)
                if not st['even']:
                    break
                pt = phrase_times(words, phrases, timing)
                per = [(e - s) / sy for v in pt.values() if v for s, e, sy in [v]]
                mean = sum(per) / len(per)
                cv = (sum((v - mean) ** 2 for v in per) / len(per)) ** 0.5 / mean * 100
                note += f' pass{p + 1}: {mean * 1000:.0f}ms±{cv:.0f}%'
                for kp, v in pt.items():
                    if v:
                        s, e, sy = v
                        cur = rates.get(kp, st['start_rate'])
                        rates[kp] = min(2.0, max(0.5, cur * (e - s) / (sy * st['syl_ms'] / 1000)))
            if st['even']:
                cues, anchors = rebuild(path, words, phrases, timing, st)
                if st.get('stretch', 1.0) != 1.0:
                    note += f' → stretched to ~{st["syl_ms"] / st["stretch"]:.0f}ms'
            else:
                if st['echo']:
                    add_echo(path)
                cues = [round(marks.get(f'L{i}', 0.0), 2) for i in range(len(words))]
                anchors = [[[wi, round(timing[(n, wi)][0], 2)] for _, _, wi, _ in offs if (n, wi) in timing] for n, offs in enumerate(words)]
            entry[slot] = fname
            entry['cues'][slot] = cues
            entry['words'][slot] = anchors
            timed = sum(len(a) for a in anchors), sum(len(o) for o in words)
            print(f'{fname:30s} {len(cues)} lines · words timed {timed[0]}/{timed[1]}{note}', flush=True)
        manifest[key_id] = entry
    manifest['_readme'] = 'สร้างโดย scripts/generate_audio_azure.py · เสียง Azure Neural "Niwat" (th-TH-NiwatNeural) · ใส่ไฟล์เสียงอัดจริงแทนได้ ดู README.md'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print('manifest:', man_path)

if __name__ == '__main__':
    main()
