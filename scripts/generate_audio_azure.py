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
import azure.cognitiveservices.speech as speechsdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'www', 'audio')
VOICE = 'th-TH-NiwatNeural'

# rate: SSML prosody rate · pitch: small lowering for a calm tone · gap: pause after each line
STYLES = {
    'lay':   dict(rate='-22%', pitch='-4%', gap=600, echo=False),
    'monk':  dict(rate='-38%', pitch='-6%', gap=900, echo=True),
    'topic': dict(rate='-8%',  pitch='-2%', gap=700, echo=False),
}

def load_texts():
    js = subprocess.run(['node', os.path.join(ROOT, 'scripts', 'export-texts.mjs')], capture_output=True, text=True, check=True)
    return json.loads(js.stdout)

def spoken_text(line):
    t = line.replace('ฯ', '')
    t = re.sub(r'[;:()"“”]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def build_ssml(lines, st):
    """Returns ssml and, for each spoken line, the list of (offset in ssml, word index)."""
    head = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="th-TH">'
            f'<voice name="{VOICE}"><prosody rate="{st["rate"]}" pitch="{st["pitch"]}">'
            f'<break time="300ms"/>')
    ssml, words = head, []
    n = 0
    for line in lines:
        if line.startswith('~'):
            continue
        ssml += f'<bookmark mark="L{n}"/>'
        offs = []
        for wi, m in enumerate(re.finditer(r'\S+', line)):
            tok = spoken_text(m.group())
            if not tok:
                continue
            if offs:
                ssml += ' '
            start = len(ssml)
            ssml += escape(tok)
            offs.append((start, len(ssml), wi))
        ssml += f'<break time="{st["gap"]}ms"/>'
        words.append(offs)
        n += 1
    ssml += '</prosody></voice></speak>'
    return ssml, words

def synthesize(cfg, ssml, out_mp3):
    audio_cfg = speechsdk.audio.AudioOutputConfig(filename=out_mp3)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
    marks, bounds = {}, []
    synth.bookmark_reached.connect(lambda e: marks.__setitem__(e.text, e.audio_offset / 1e7))
    synth.synthesis_word_boundary.connect(lambda e: bounds.append((e.text_offset, e.audio_offset / 1e7)))
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

def add_echo(mp3):
    tmp = mp3 + '.tmp.mp3'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-af',
                    'aecho=0.8:0.55:70|140:0.22|0.12,loudnorm=I=-18:TP=-2:LRA=7',
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
            ssml, words = build_ssml(lines, st)
            slot = 'lay' if sname == 'topic' else sname
            fname = (key_id.replace(':', '-') + '.mp3') if sname == 'topic' else f'{key_id}-{sname}.mp3'
            path = os.path.join(OUT, fname)
            marks, bounds = synthesize(cfg, ssml, path)
            if st['echo']:
                add_echo(path)
            cues = [round(marks.get(f'L{i}', 0.0), 2) for i in range(len(words))]
            # Azure splits Thai into its own words: give each of our space-separated tokens
            # the time of the first boundary that falls inside it.
            anchors = []
            for offs in words:
                line_a = []
                for a, b, wi in offs:
                    ts = [t for off, t in bounds if a <= off < b]
                    if ts:
                        line_a.append([wi, round(min(ts), 2)])
                anchors.append(line_a)
            entry[slot] = fname
            entry['cues'][slot] = cues
            entry['words'][slot] = anchors
            matched = sum(len(a) for a in anchors), sum(len(o) for o in words)
            print(f'{fname:32s} {len(cues)} lines · words timed {matched[0]}/{matched[1]}', flush=True)
        manifest[key_id] = entry
    manifest['_readme'] = 'สร้างโดย scripts/generate_audio_azure.py · เสียง Azure Neural "Niwat" (th-TH-NiwatNeural) · ใส่ไฟล์เสียงอัดจริงแทนได้ ดู README.md'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print('manifest:', man_path)

if __name__ == '__main__':
    main()
