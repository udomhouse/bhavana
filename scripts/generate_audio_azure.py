#!/usr/bin/env python3
"""Generate all chant and dhamma audio with Azure Speech, Thai voice "Niwat" (th-TH-NiwatNeural).

One voice for the whole app. Styles differ only in pace, pauses and a little room echo.
Azure reports the exact time of every line (bookmarks) and every word (word boundaries),
so the app highlights the word being spoken.

Setup:
    pip install azure-cognitiveservices-speech
    export AZURE_SPEECH_KEY=...            # Keys and Endpoint › KEY 1
    export AZURE_SPEECH_REGION=southeastasia
Run:
    python3 scripts/generate_audio_azure.py                      # everything
    python3 scripts/generate_audio_azure.py --only itipiso,ariyasacca
Needs node (reads texts from www/app.js) and ffmpeg (echo for the slow style).
Output: www/audio/*.mp3 and www/audio/manifest.json (same format the app already reads).
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from xml.sax.saxutils import escape
import numpy as np
import azure.cognitiveservices.speech as speechsdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'www', 'audio')
VOICE = 'th-TH-NiwatNeural'

# Chants use "even" timing: every syllable gets the same length (syl_ms), words are separated by a
# short breath (word_ms), and the voice pauses only at commas (comma_ms) and line ends (gap).
# Azure is asked to synthesize, the real length of each word is measured, and each word's speed is
# corrected (passes) until the syllables are evenly spaced — the steady pulse people chant along to.
# Dhamma talks keep natural speech. pitch: small lowering for a calm tone.
STYLES = {
    'lay':   dict(even=True, syl_ms=330, word_ms=70,  comma_ms=420, gap=850,  pitch='-4%', echo=False, start_rate=0.80),
    # Azure cannot go slower than half speed, so the slow style is synthesized evenly at ~330 ms and then
    # time-stretched (pitch unchanged) to ~435 ms per syllable; cue and word times are scaled to match.
    'monk':  dict(even=True, syl_ms=330, word_ms=80,  comma_ms=450, gap=950,  pitch='-6%', echo=True,  start_rate=0.75, stretch=0.76),
    'topic': dict(even=False, rate='-8%', pitch='-2%', gap=700, echo=False),
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
    t = line.replace('ฯ', '')
    t = re.sub(r'[;:()"“”]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def build_ssml(lines, st, rates=None):
    """Returns ssml and, for each spoken line, [(start, end, word index, syllables)] offsets into the ssml.
    rates: {(line, word index): speed multiplier} for even timing."""
    if st['even']:
        head = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
                f'<voice name="{VOICE}"><prosody pitch="{st["pitch"]}"><break time="300ms"/>')
    else:
        head = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
                f'<voice name="{VOICE}"><prosody rate="{st["rate"]}" pitch="{st["pitch"]}"><break time="300ms"/>')
    ssml, words, n = head, [], 0
    for line in lines:
        if line.startswith('~'):
            continue
        ssml += f'<bookmark mark="L{n}"/>'
        offs = []
        toks = [(wi, m.group()) for wi, m in enumerate(re.finditer(r'\S+', line))]
        for k, (wi, raw) in enumerate(toks):
            tok = spoken_text(raw)
            if st['even']:
                tok = tok.strip(',.')
            if not tok:
                continue
            if st['even']:
                r = (rates or {}).get((n, wi), st['start_rate'])
                ssml += f'<prosody rate="{r:.3f}">'
                start = len(ssml); ssml += escape(tok); end = len(ssml)
                ssml += '</prosody>'
                if k < len(toks) - 1:
                    pause = st['comma_ms'] if raw.endswith(',') else st['word_ms']
                    ssml += f'<break time="{pause}ms"/>'
            else:
                if offs:
                    ssml += ' '
                start = len(ssml); ssml += escape(tok); end = len(ssml)
            offs.append((start, end, wi, syllables(tok)))
        ssml += f'<break time="{st["gap"]}ms"/>'
        words.append(offs)
        n += 1
    ssml += '</prosody></voice></speak>'
    return ssml, words

def measure(words, bounds):
    """Per word: (start time, spoken length) from Azure word boundaries falling inside it."""
    out = {}
    for n, offs in enumerate(words):
        for a, b, wi, _ in offs:
            seg = [(t, t + d) for off, t, d in bounds if a <= off < b]
            if seg:
                out[(n, wi)] = (min(x for x, _ in seg), max(y for _, y in seg) - min(x for x, _ in seg))
    return out

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

SR = 24000

def rebuild(path, words, timing, st, spoken):
    """Azure leaves long, uneven silences between separately-paced words. Cut each word out using its
    boundary times and lay them back to back with our own fixed pauses: word_ms between words,
    comma_ms after a comma, gap after the line. Returns (cues, anchors) in the rebuilt audio."""
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    sil = lambda ms: np.zeros(int(SR * ms / 1000), dtype=np.float32)
    fade = int(0.008 * SR)
    out, t, cues, anchors = [sil(300)], 0.3, [], []
    for n, offs in enumerate(words):
        cues.append(round(t, 2))
        toks = spoken[n].split()
        known = [(wi, *timing[(n, wi)]) for _, _, wi, _ in offs if (n, wi) in timing]
        if len(known) < len(offs) and known:   # a word without timing: keep the whole line as one piece
            s0 = min(k[1] for k in known); e0 = max(k[1] + k[2] for k in known)
            known = [(known[0][0], s0, e0 - s0)]
        line_a = []
        for j, (wi, s0, d) in enumerate(known):
            seg = x[max(0, int((s0 - 0.025) * SR)): min(len(x), int((s0 + d + 0.05) * SR))].copy()
            if len(seg) > 2 * fade:
                seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            line_a.append([wi, round(t + 0.025, 2)])
            out.append(seg); t += len(seg) / SR
            ms = st['gap'] if j == len(known) - 1 else (st['comma_ms'] if toks[wi].endswith(',') else st['word_ms'])
            out.append(sil(ms)); t += ms / 1000
        anchors.append(line_a)
    y = np.concatenate(out)
    filters = []
    if st.get('stretch', 1.0) != 1.0:
        filters.append(f"atempo={st['stretch']}")
    if st['echo']:
        filters.append('aecho=0.8:0.55:70|140:0.22|0.12')
    filters.append('loudnorm=I=-18:TP=-2:LRA=7')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                    '-af', ','.join(filters), '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', path],
                   input=y.tobytes(), check=True)
    k = 1 / st.get('stretch', 1.0)
    return [round(c * k, 2) for c in cues], [[[wi, round(tt * k, 2)] for wi, tt in la] for la in anchors]

def add_echo(mp3, stretch=1.0):
    tmp = mp3 + '.tmp.mp3'
    tempo = f'atempo={stretch},' if stretch != 1.0 else ''
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-af',
                    tempo + 'aecho=0.8:0.55:70|140:0.22|0.12,loudnorm=I=-18:TP=-2:LRA=7',
                    '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', tmp], check=True)
    os.replace(tmp, mp3)

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
    cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary, 'false')

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
            rates, spread = None, ''
            for p in range(PASSES if st['even'] else 1):
                ssml, words = build_ssml(lines, st, rates)
                marks, bounds = synthesize(cfg, ssml, path)
                if not st['even']:
                    break
                timing = measure(words, bounds)
                per = [timing[(n, wi)][1] / sy for n, offs in enumerate(words) for _, _, wi, sy in offs if (n, wi) in timing]
                mean = sum(per) / len(per)
                cv = (sum((x - mean) ** 2 for x in per) / len(per)) ** 0.5 / mean * 100
                spread += f' pass{p + 1}: {mean * 1000:.0f}ms±{cv:.0f}%'
                rates = rates or {}
                for n, offs in enumerate(words):
                    for _, _, wi, sy in offs:
                        if (n, wi) in timing:
                            cur = rates.get((n, wi), st['start_rate'])
                            want = sy * st['syl_ms'] / 1000
                            rates[(n, wi)] = min(2.0, max(0.5, cur * timing[(n, wi)][1] / want))
            timing = measure(words, bounds)
            if st['even']:
                spoken = [l for l in lines if not l.startswith('~')]
                cues, anchors = rebuild(path, words, timing, st, spoken)
                k = 1 / st.get('stretch', 1.0)
                if k != 1:
                    spread += f' → stretched to ~{st["syl_ms"] * k:.0f}ms'
            else:
                if st['echo']:
                    add_echo(path)
                cues = [round(marks.get(f'L{i}', 0.0), 2) for i in range(len(words))]
                anchors = [[[wi, round(timing[(n, wi)][0], 2)] for _, _, wi, _ in offs if (n, wi) in timing] for n, offs in enumerate(words)]
            entry[slot] = fname
            entry['cues'][slot] = cues
            entry['words'][slot] = anchors
            matched = sum(len(a) for a in anchors), sum(len(o) for o in words)
            print(f'{fname:30s} {len(cues)} lines · words timed {matched[0]}/{matched[1]}{spread}', flush=True)
        manifest[key_id] = entry
    manifest['_readme'] = 'สร้างโดย scripts/generate_audio_azure.py · เสียง Azure Neural "Niwat" (th-TH-NiwatNeural) · ใส่ไฟล์เสียงอัดจริงแทนได้ ดู README.md'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print('manifest:', man_path)

if __name__ == '__main__':
    main()
