#!/usr/bin/env python3
"""Generate chanting audio for every chant and dhamma topic with an offline Thai TTS model.

Model: Meta MMS-TTS Thai (VITS), via sherpa-onnx. License: CC-BY-NC 4.0 (non-commercial use only).
Output: www/audio/<id>-lay.mp3, www/audio/<id>-monk.mp3, topic-<id>.mp3 and www/audio/manifest.json
with per-line cue times so the app highlights the line being chanted.

Setup (once):
    pip install sherpa-onnx soundfile numpy
    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-tha.tar.bz2
    tar xjf vits-mms-tha.tar.bz2
Run:
    python3 scripts/generate_audio.py --model vits-mms-tha            # everything
    python3 scripts/generate_audio.py --model vits-mms-tha --only itipiso,ngoenlan
Needs node (to read the texts from www/app.js) and ffmpeg (to encode mp3).
"""
import argparse, json, os, re, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
import sherpa_onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'www', 'audio')

# voice style: TTS speed (lower = slower), silence after each line (s), pitch factor (<1 = deeper)
VOICES = {
    'lay':   dict(speed=0.82, gap=0.55, pitch=1.0,  echo=False),
    'monk':  dict(speed=0.70, gap=0.75, pitch=0.80, echo=True),
    'topic': dict(speed=0.95, gap=0.70, pitch=1.0,  echo=False),
}

THAI_DIGITS = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
UNITS = ['', 'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า']
PLACES = ['', 'สิบ', 'ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน']

def num_th(n: int) -> str:
    if n == 0:
        return 'ศูนย์'
    s, digits = '', str(n)[::-1]
    for i, d in enumerate(map(int, digits)):
        if d == 0:
            continue
        if i == 0 and d == 1 and len(digits) > 1:
            w = 'เอ็ด'
        elif i == 1 and d == 2:
            w = 'ยี่'
        elif i == 1 and d == 1:
            w = ''
        else:
            w = UNITS[d]
        s = w + PLACES[i] + s
    return s

def normalize(text: str) -> str:
    t = text.translate(THAI_DIGITS)
    t = re.sub(r'\d+', lambda m: num_th(int(m.group())), t)
    t = re.sub(r'(\S+)\s*ๆ', r'\1 \1', t)          # mai yamok: repeat the word
    t = t.replace('ำ', 'ํา').replace('ฯ', '').replace('ฺ', '')
    t = re.sub(r'[,.;:()"“”!?·]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def load_texts():
    js = subprocess.run(['node', os.path.join(ROOT, 'scripts', 'export-texts.mjs')], capture_output=True, text=True, check=True)
    return json.loads(js.stdout)

def synth_lines(tts, lines, v):
    sr = tts.sample_rate
    gap = np.zeros(int(sr * v['gap']), dtype=np.float32)
    chunks, cues, t = [np.zeros(int(sr * 0.3), dtype=np.float32)], [], 0.3
    for line in lines:
        if line.startswith('~'):
            continue
        audio = tts.generate(normalize(line), sid=0, speed=v['speed'])
        samples = np.asarray(audio.samples, dtype=np.float32)
        cues.append(round(t, 2))
        chunks += [samples, gap]
        t += (len(samples) + len(gap)) / sr
    return np.concatenate(chunks), sr, cues

def encode(wav_path, mp3_path, v, sr):
    filters = []
    if v['pitch'] != 1.0:   # deeper voice, same tempo, so the cue times stay valid
        filters += [f"asetrate={int(sr * v['pitch'])}", f'aresample={sr}', f"atempo={1 / v['pitch']:.4f}"]
    if v['echo']:           # a little hall reverb
        filters += ['aecho=0.8:0.6:60|120:0.25|0.15']
    filters += ['loudnorm=I=-18:TP=-2:LRA=11']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav_path, '-af', ','.join(filters),
                    '-ac', '1', '-ar', '22050', '-c:a', 'libmp3lame', '-b:a', '40k', mp3_path], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='folder of vits-mms-tha (model.onnx, tokens.txt)')
    ap.add_argument('--only', default='', help='comma-separated ids to regenerate')
    args = ap.parse_args()
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=os.path.join(args.model, 'model.onnx'),
                                                    tokens=os.path.join(args.model, 'tokens.txt')),
        num_threads=os.cpu_count() or 2)))
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, 'manifest.json')
    manifest = json.load(open(man_path, encoding='utf8')) if os.path.exists(man_path) else {}
    only = set(filter(None, args.only.split(',')))
    data = load_texts()
    jobs = [(c['id'], c['lines'], ['lay', 'monk']) for c in data['chants']] + \
           [(t['id'], t['lines'], ['topic']) for t in data['topics']]
    for key, lines, voices in jobs:
        if only and key not in only and key.replace('topic:', '') not in only:
            continue
        entry = {'cues': {}}
        for vname in voices:
            v = VOICES[vname]
            samples, sr, cues = synth_lines(tts, lines, v)
            slot = 'lay' if vname == 'topic' else vname
            fname = (key.replace(':', '-') + '.mp3') if vname == 'topic' else f'{key}-{vname}.mp3'
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                sf.write(tmp.name, samples, sr)
                encode(tmp.name, os.path.join(OUT, fname), v, sr)
            entry[slot] = fname
            entry['cues'][slot] = cues
            print(f'{fname:32s} {len(samples) / sr:6.1f}s  {len(cues)} lines', flush=True)
        manifest[key] = entry
    manifest['_readme'] = 'สร้างโดย scripts/generate_audio.py (เสียงสังเคราะห์ MMS-TTS Thai, CC-BY-NC 4.0) · ใส่ไฟล์เสียงอัดจริงแทนได้ ดู README.md'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print('manifest:', man_path)

if __name__ == '__main__':
    main()
