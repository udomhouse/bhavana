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
CACHE = os.path.join(ROOT, 'scripts', '.tts-cache-pcm')   # takes as delivered (uncompressed WAV)
TIGHTEN = (0.16, 0.07)     # inside a phrase, breaths longer than 0.16 s become 0.07 s

# Chant styles ("even"): syl_ms target length per syllable · comma_ms pause between phrases of a line ·
# gap pause at line end · stretch slows the slow style a little afterwards (pitch kept) · echo room echo.
# The voice's natural pace is close to 300 ms per syllable, so the normal style barely changes its speed.
STYLES = {
    'lay':   dict(even=True, syl_ms=280, comma_ms=700, gap=1000, echo=False, start_rate=1.0),
    'monk':  dict(even=True, syl_ms=280, comma_ms=650, gap=880,  echo=True,  start_rate=1.0, stretch=0.88),
    'topic': dict(even=False, rate=0.95, comma_ms=320, gap=900, echo=False),   # a touch slower: listeners found some words rushed
}
VOWELS = set('ะาิีึืุูัเแโไใำ')

# How a word should sound when the written form misleads the voice (display text is unchanged).
PRONOUNCE = {
    'ฆราวาสธรรม': 'คราวาดทำ',
    # listener's review page, round 4 (Nov 15)
    'โมคคัลลานัญจะ': 'โมกคันลานันจะ',
    'ติณณันนัมปิ': 'ตินนันนัมปิ',
    'พาหิรัชฌัตตุปัททะวา': 'พาหิรัด ชัดตุปัดทะวา',
    'สัททัง': 'สัดทัง',
    'พายัพรัสมิง': 'พายับ รัดสะหมิง',
    'ปฐวีรัสมิง': 'ปะถะวี รัดสะหมิง',
    'เมตตัมพุเสกะวิธินา': 'เมตตัม พุเส กะวิธินา',
    'สัพพะโลกัสมิง': 'สัพพะโลกัด-สะ-หมิง',
    'อัพยาปัชโฌ': 'อับพะยาปัดโช',
    'เอตัมมังคะละมุตตะมัง': 'เอ-ตำ-มัง-คะ-ละ-มุด-ตะ-มัง',
    'อุเทตะยัญจักขุมา': 'อุเทตะ ยันจักขุมา',
    'พุทธะมะอะอุ': 'พุดทะ, มะ, อะ, อุ',
    'พุทธัสสะ': 'พุดทัดสะ',
    'สัมปะติจฉามิ': 'สำปะติดฉามิ',
    'อะหังปิตัง': 'อะหัง ปิตัง',
    'วูปะสะโม': 'วูปะ สะโม',
    'อะทินนาทานา': 'อะทินนา ทานา',
    'นิโรธเป็น': 'นิโรด เป็น',
    'เพียร': 'เพียน',
    'อิสิปตนมฤคทายวัน': 'อิสิปะตะนะมะริคคะทายะวัน',
    'สมุทัย': 'สะ-หมุ-ทัย',
    'สะหัสสะมะภินิมมิตะสาวุธันตัง': 'สะหัสสะมะภินิมมิตะ สาวุธันตัง',
    'ขอส่วนบุญนี้': 'ขอส่วนบุญนี้ ',      # the Thai dedication lines: pause where the listener reads one
    'จงสำเร็จแก่บิดามารดา': 'จงสำเร็จแก่ บิดา มาน ดา',
    'ทั้งหลายทั้งปวงมีความสุข': 'ทั้งหลายทั้งปวง มีความสุข',
    'หิตวานะเนกะวิวิธานิ': 'หิตวานะเนกะ วิวิธานิ',
    'หิงสิ': 'หิง-สิ',
    'มุนินโท': 'มุ-นิน-โท',
    'ปัญจะสัตตะ': 'ปัญจะ สัตตะ',
    'เสนียดจัญไร': 'สะเหนียด จันไร',
    'ไม่มีเวร': 'ไม่มีเวน',
    'นันทิวัฑฒะโน': 'นันทิ-วัด-ทะ-โน',
    'สัพเพ สัตตา': 'สับ-เพ สัด-ตา',
    'อาฬวกสูตร': 'อาละวะกะสูด',
    'อาฬวก': 'อาละวะกะ',
    'โพชฌังโค': 'โพด-ชัง-โค',
    'สะมาคะตัตโต': 'สะมาคะตัด-โต',
    'อะนะนุสสุเตสุ': 'อะนะนุด-สุเตสุ',
    'ญาณัญจะ': 'ยานัน-จะ',
    'วิสุทธิชุติมิทธิพะกาภิธานัง': 'วิสุด-ทิ-ชุ-ติ-มิด-ทิ-พะ-กา-พิ-ทา-นัง',
    'สักยานัง': 'สัก-กะ-ยา-นัง',
    'จะตุปปะเทหิ': 'จะตุบปะเทฮิ',     # listener: จะ-ตุบ-ปะ-เท-ฮิ
    'ทุติยัมปิ': 'ทุ-ติ-ยัม-ปิ',
    'หีนา': 'ฮีนา',
    'กาตุญจะ': 'กาตุนจะ',
    'กุสะลา': 'กุ สะลา',
    'เหตุโปวะ': 'เห-ตุ-โป-วะ',
    'มหาสัปปุริสะสาวัง': 'มะหา สับปุริสะ สาวัง',
    'สัปปุริสะสาวัง': 'สับปุริสะ สาวัง',
    'โหนตุ': 'โหน-ตุ',
    'โมกขัง': 'โมก ขัง',
    'โสตถี': 'โสดถี',            # listener's pick (sample 3)
    'สิกขาปะทัง': 'สิก-ขา-ปะ-ทัง',
    'ปิฏฐิภาคัสมิง': 'ปิดถิ พาคัดสะมิง',
    'สังฆัสสาหัสมิ': 'สังคัดสา หัดสะมิ',
    # listener's review page, Nov 2026 (longer words first: a plain replace runs in this order)
    'อัพภุคคะโต': 'อับพุคคะโต',
    'พุทธัสสาหัสมิ': 'พุดทัสสาหัดสะมิ',
    'โกณฑัญโญ': 'โกนทันโย',
    'โมคคัลลาโน': 'โมกคันลาโน',
    'ขันธะโมระปะริตตัญจะ': 'ขันธะโมระปะริดตันจะ',
    'สัคคะโมกขะทัง': 'สักคะโมกขะทัง',
    'ทุกโข': 'ทุก โข',
    'ปะหีนันติ': 'ปะฮีนันติ',
    'โลกัสมินติ': 'โลกัดสะมินติ',
    'สัททะมะนุสสาเวสุง': 'สัดทะมะนุดสาเวสุง',
    'เทวะมะนุสสานัง': 'เทวะมะนุดสานัง',
    'มาราติเรกะมะภิยุชฌิตะสัพพะรัตติง': 'มาราติเรกะมะภิยุดชิตะสัพพะรัดติง',
    'กัตวานะ': 'กัดตะวานะ',
    'ทุคคาหะทิฏฐิภุชะเคนะ': 'ทุกคาหะทิดถิพุชะเคนะ',
    'มานะสัมภาวะเย': 'มานะสำพาวะเย',
    'ธัมมัง': 'ทำมัง',
    'บิดามารดา': 'บิดา มาน ดา',
    'มะหาเถรัง': 'มะหาเถ รัง',
    'ฤๅ': 'รือ',
    'ชานามิ': 'ชา นามิ',
    'ชีวิตัง': 'ชีวิ ตัง',
    'ฉุฑโฑ': 'ฉุดทะโท',
    'ตะติยัมปิ': 'ตะติ ยัมปิ',
    'วิริเยสุ': 'วิริ เยสุ',
    'นะมามิ': 'นะ มามิ',
    'ฬ': 'ล',                   # จักกะวาฬะ → จัก-กะ-วา-ละ
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
    'หุลู': 'หุ่ลู',          # listener's pick (sample E): short low หุ, longer ลู
    'เวทะนาขันโธ': 'เวดทะนา ขันโธ',
    # listener's corrections, round 3 (words with ฏฐ first, before the general ฏฐ rule)
    'ปะติฏฐาสิ': 'ปะติดทาสิ',
    'เอหิปัสสิโก': 'เอหิ ปัดสิโก',
    'หีโน': 'ฮีโน',
    'ชาติ': 'ชา ติ',
    'ชาตุ': 'ชา ตุ',
    'อายัสมะโต': 'อายัด สะมะโต',
    'เทวา': 'เท วา',
    'โลกัสมิง': 'โลกัดสะ มิง',
    'ขุททัง': 'ขุดทัง',
    'พรัหม': 'พรัมม',          # พรัหมัง → พรัม-มัง (and พรัหมะ…, like พรหมา)
    'ฤา': 'รือ',
    'อัพยาปัชฌา': 'อับ-พะ-ยา-ปัด-ชา',
    'ฏฐ': 'ดถ',               # ปะติฏฐาสิ → ปะ-ติด-ถา-สิ (must come before ฏ)
    'ฏ': 'ต',                  # สุปะฏิปันโน → สุ-ปะ-ติ-ปัน-โน
    'โหตุ': 'โห, ตุ',          # listener's pick (sample 2) over hyphen / space / โหะ
    'มารดา': 'มาน ดา',
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
# Phrases the listener approved from a sample: same text, rate and (empty) salt → the very same cached take.
PINNED = {'หุ่ลู หุ่ลู หุ่ลู สะวาหายะ': (0.78, ''),   # text → (rate, salt)
          'อะนุตตะรัง สิระสา นะ มามิ': (0.89, '|p1'),
          'เมตตัม พุเส กะวิธินา ชิตะวา มุ-นิน-โท': (0.85, '|p1'),
          'สัพพะธัมมานุภาเวนะ สะทา โสดถี ภะวันตุ เต': (0.85, '|p2'),
          'อิมินา ปุญญะกัมเมนะ อิมินา อุททิเสนะ จะ': (0.89, '|p1'),
          'อิทธิบาท': (0.95, '|p3')}
SPLIT_BEFORE = {'หุลู'}   # start a new phrase here even without a comma, so it can be paced on its own
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

# whole words only (a plain replace would also hit them inside longer words)
EXACT = {'ทมะ': 'ทะมะ', 'นิโรธ': 'นิโรท'}   # (หิ→ฮิ dropped: the listener wants a plain หิ)   # นะ หิ ชาตุ → นะ ฮิ ชา ตุ

def spoken_text(t):
    core = t.strip(',.;:')
    if core in EXACT:
        t = t.replace(core, EXACT[core])
    t = t.replace('ฯ', '').replace('ฺ', '').replace('-', ' ')   # phinthu confuses the voice; ยะ-ธา-พุท is read per syllable
    t = re.sub(r'[;:()"“”]', ' ', t)
    t = re.sub(r'(เล็ก|บ่อย|ค่อย|เฉย|ช้า|ยาว|นาน|ดี) ๆ', r'\1 \1', t)   # spaced ไม้ยมก in the talks: เรื่องเล็ก ๆ → เรื่องเล็ก เล็ก
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
                if raw.endswith(',') or wi == len(toks) - 1 or (toks[wi + 1] in SPLIT_BEFORE and raw not in SPLIT_BEFORE):
                    words = [spoken_text(toks[i]).strip(',.') for i in cur]
                    ws = [w for w in words if w]
                    # a run of one-syllable words (โส โส สะ สะ อะ …): the voice drops some unless each is set apart
                    text = (', ' if len(ws) >= 6 and all(syllables(w) <= 1 for w in ws) else ' ').join(ws)
                    prose = is_thai_prose(text)
                    weights = [max(1, len(w) // 3) for w in words] if prose else [syllables(w) for w in words]
                    factor = max([f for k, f in SLOWER.items() if k in text] or [1.0])
                    units.append((n, p, cur, weights, text, prose, factor))
                    p, cur = p + 1, []
        else:
            # one unit per clause (the text's spaces): the voice then cannot pause or drop words in odd places
            # inside a long paragraph; clauses shorter than 8 letters join the next one.
            p, cur = 0, []
            for wi, raw in enumerate(toks):
                cur.append(wi)
                if (len(''.join(toks[i] for i in cur)) >= 8 and toks[min(wi + 1, len(toks) - 1)] != 'ๆ') or wi == len(toks) - 1:
                    units.append((n, p, cur, [max(1, len(toks[i]) // 3) for i in cur],
                                  spoken_text(' '.join(toks[i] for i in cur)), True, 1.0))
                    p, cur = p + 1, []
        n += 1
    return units, n

def decode(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def synthesize(cfg, ssml):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
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

def tighten(seg, limits=TIGHTEN):
    """Shorten the voice's own breaths inside one phrase so its words flow together."""
    out, last = [], 0
    keep = int(limits[1] * SR)
    for a, b in silences(seg, limits[0]):
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

def trim(x, lead=0.06, tail=0.10):
    """Cut leading and trailing silence, keeping a margin so soft onsets (ส, ข, ฉ) and the
    last syllable's release are not clipped."""
    runs = silences(x, 0.02)
    a = runs[0][1] if runs and runs[0][0] == 0 else 0
    b = runs[-1][0] if runs and runs[-1][1] >= len(x) else len(x)
    a = max(0, a - int(lead * SR)); b = min(len(x), b + int(tail * SR))
    return x[a:b] if b > a else x

def speak(cfg, text, rate, salt=''):
    """One phrase → audio (cached by voice, rate and text; a salt asks for fresh takes)."""
    key = hashlib.sha1(f'{VOICE}|{rate:.3f}|{text}|raw{salt}'.encode()).hexdigest()
    path = os.path.join(CACHE, key + '.f32')
    if os.path.exists(path):   # the cache keeps the untouched take; trimming happens on load
        return trim(np.fromfile(path, dtype=np.float32))
    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="th-TH">'
            f'<voice name="{VOICE}"><prosody rate="{rate:.3f}">{escape(text)}</prosody></voice></speak>')
    raw = synthesize(cfg, ssml)
    os.makedirs(CACHE, exist_ok=True)
    raw.astype(np.float32).tofile(path)
    return trim(raw)

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

def sing_measure(x):
    """(pitch range in semitones, longest held note in s) — a sung take swings in pitch like a melody
    and holds vowels on one note; reading does neither. The phrase-final syllable may linger, so the
    last 0.6 s is left out of the held-note measure."""
    n, h = int(0.04 * SR), int(0.01 * SR)
    floor = 0.12 * float(np.sqrt((x ** 2).mean())) if len(x) else 0
    f = []
    for s in range(0, len(x) - n, h):
        fr = x[s:s + n]
        if np.sqrt((fr ** 2).mean()) < floor:
            f.append(0); continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, 'full')[n - 1:]
        lo, hi = SR // 350, SR // 60
        k = lo + int(np.argmax(ac[lo:hi]))
        f.append(SR / k if ac[0] > 0 and ac[k] / ac[0] > .5 else 0)
    v = [p for p in f if p]
    if len(v) < 10:
        return 0.0, 0.0
    rng = 12 * np.log2(np.percentile(v, 95) / np.percentile(v, 5))
    body = f[:max(0, len(f) - 60)]
    best = run = 0; prev = 0
    for p in body:
        run = run + 1 if p and prev and abs(p - prev) / prev < 0.04 else 0
        best = max(best, run); prev = p
    return float(rng), best * h / SR

def music_bed(x):
    """The voice model sometimes lays music under a long take. Speech leaves near-silent gaps
    every few seconds; music fills them. Returns the smallest share of near-silent 20 ms frames in
    any 4 s window (speech: 0.1-0.35, with music: ~0)."""
    h, w = int(0.02 * SR), 200
    if len(x) < 4.5 * SR:
        return 1.0
    r = np.array([np.sqrt((x[i:i + h] ** 2).mean()) for i in range(0, len(x) - h, h)])
    md = float(np.median(r[r > 0])) if (r > 0).any() else 0
    q = (r < 0.03 * md).astype(float)
    return min(q[i:i + w].mean() for i in range(0, max(1, len(q) - w), 50))

MUSIC_MIN = 0.06
GLITCH_S = 0.12
SING_RANGE, SING_HOLD = 16.0, 0.35
JITTER = [0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04]
glitch_log = []
UNIT_SALT = {}   # unit index → salt for fresh takes (set per chant from the listener's reports)

# Chants read like plain speech: each line in one go at the voice's own pace, with stricter
# take selection (the listener picked this style for ศีล ๕ by ear: store/voice-test/sila5-A-whole-line.mp3)
NATURAL = {'sila5'}
STRICT = None   # set while rendering a NATURAL chant: (glitch, hold, range, extra jitter)

def final_hold(x):
    """Seconds the last syllable stays on one pitch (within 3%). Over ~0.15 s it is heard as a
    ringing or musical note at the end of the phrase (listener reports, round 4)."""
    n, h = int(0.04 * SR), int(0.01 * SR)
    floor = 0.12 * float(np.sqrt((x ** 2).mean())) if len(x) else 0
    f = []
    for s in range(max(0, len(x) - int(1.2 * SR)), len(x) - n, h):
        fr = x[s:s + n]
        if np.sqrt((fr ** 2).mean()) < floor:
            f.append(0); continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, 'full')[n - 1:]
        lo, hi = SR // 350, SR // 60
        k = lo + int(np.argmax(ac[lo:hi]))
        f.append(SR / k if ac[0] > 0 and ac[k] / ac[0] > .5 else 0)
    while f and f[-1] == 0:
        f.pop()
    run = []
    for p in reversed(f):
        if not p or (run and abs(p - np.median(run)) / np.median(run) > 0.03):
            break
        run.append(p)
    return len(run) * 0.01

FINAL_HOLD = 0.14

def speak_clean(cfg, text, rate, salt='', prose=False):
    """speak(), retried at a hair's different speed while the take contains a pure-tone glitch."""
    best = None
    gl, sh, sr, extra = STRICT or (GLITCH_S - 0.01, SING_HOLD, SING_RANGE, [])
    if prose:
        sh = 0.28   # talks: a drawn-out syllable sounds odd (listener: ความ… นานไป)
    for d in JITTER + extra:
        x = speak(cfg, text, round(rate + d, 2), salt)
        g = tone_glitch(x)
        rng, hold = sing_measure(x)
        fh = final_hold(x)
        bad = (10 * max(0, g - gl) + 3 * max(0, hold - sh) + 0.1 * max(0, rng - sr)
               + 20 * max(0, MUSIC_MIN - music_bed(x)) + 5 * max(0, FIRST_MIN - first_syllable(x))
               + 5 * max(0, fh - FINAL_HOLD))
        if best is None or bad < best[0]:
            best = (bad, x, g, rng, hold)
        if bad == 0:
            if d: glitch_log.append(f'fixed {text[:24]}')
            return x
    glitch_log.append(f'STILL tone {best[2]:.2f}s range {best[3]:.0f}st hold {best[4]:.2f}s {text[:24]}')
    return best[1]

def frames(x, n=0.04, h=0.01):
    """(rms, share of energy in the strongest spectral peak) per 10 ms frame."""
    n, h = int(n * SR), int(h * SR)
    win = np.hanning(n).astype(np.float32)
    out = []
    for s in range(0, max(1, len(x) - n), h):
        fr = x[s:s + n]
        if len(fr) < n:
            break
        sp = np.abs(np.fft.rfft(fr * win)) ** 2
        k = int(np.argmax(sp[3:])) + 3
        out.append((float(np.sqrt((fr ** 2).mean())), float(sp[max(0, k - 2):k + 3].sum() / (sp.sum() + 1e-12))))
    return out

HUM_KEEP = 0.12
def trim_tail_hum(x):
    """The voice ends phrases on a long, nearly pure hum of the final ง/น/ม (heard as ringing).
    Keep at most HUM_KEEP s of it and fade out."""
    fr = frames(x)
    if len(fr) < 20:
        return x
    mx = max(r for r, _ in fr)
    last = max((i for i, (r, _) in enumerate(fr) if r > 0.05 * mx), default=None)
    if last is None:
        return x
    i = last
    while i > 0 and fr[i - 1][1] > 0.55 and fr[i - 1][0] > 0.03 * mx:
        i -= 1
    run = (last - i) * 0.01
    if run > HUM_KEEP + 0.04:
        cut = int((i * 0.01 + HUM_KEEP) * SR)
    else:
        # no long hum: still cut the echo-like decay once the last syllable has faded to a fifth
        j = last
        top = max(r for r, _ in fr[max(0, last - 30):last + 1])
        while j > 0 and fr[j][0] < 0.2 * top:
            j -= 1
        cut = int((j + 4) * 0.01 * SR)   # frame start + window
        if cut >= len(x) - int(0.03 * SR):
            return x
    fade = int(0.06 * SR)
    y = x[:cut + fade].copy()
    y[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32) ** 2
    return y

def first_syllable(x):
    """Seconds from the onset to the first clear loudness dip: the voice sometimes clips the first
    syllable of a phrase (heard as a missing word)."""
    h = int(0.01 * SR)
    env = np.array([np.sqrt((x[i:i + h] ** 2).mean()) for i in range(0, len(x) - h, h)])
    if len(env) < 10:
        return 1.0
    pk = env.max(); on = int(np.argmax(env > 0.05 * pk)); top = 0
    for i in range(on, len(env)):
        top = max(top, env[i])
        if env[i] < 0.3 * top and i - on > 3:
            return (i - on) * 0.01
    return 1.0

FIRST_MIN = 0.11

def render_units(cfg, units, st, rates, idx=None):
    """Synthesize every unit on its own and return one audio array per unit."""
    def one(i):
        u = units[i]
        base = st.get('start_rate', st.get('rate', 1.0))
        r = round(base / u[6] if u[5] else rates.get(i, base / u[6]), 2)
        salt = UNIT_SALT.get(i, '')
        if u[4] in PINNED:
            r, salt = PINNED[u[4]]
        x = trim_tail_hum(speak_clean(cfg, u[4], r, salt, u[5]))
        # (Syllable-by-syllable time-stretching was tried and removed: listeners heard the voice wobble.)
        if st['even'] and not u[5]:
            return tighten(x)
        return tighten(x, (0.25, 0.12))   # Thai prose, in talks and inside chants alike: no long pauses mid-phrase
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(one, range(len(units)) if idx is None else idx))

def encode(y, path, st):
    filters = []
    if st.get('stretch', 1.0) != 1.0:
        filters.append(f"atempo={st['stretch']}")
    if st['echo']:
        filters.append('aecho=0.8:0.55:70|140:0.22|0.12')
    # Loudness in two passes with one fixed gain (linear): single-pass loudnorm rides the gain and the voice wavers.
    pcm = y.astype(np.float32).tobytes()
    inp = ['-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-']
    pre = ','.join(filters) + ',' if filters else ''
    log = subprocess.run(['ffmpeg', '-hide_banner', '-nostats'] + inp + ['-af', pre + 'loudnorm=I=-18:TP=-2:LRA=11:print_format=json', '-f', 'null', '-'],
                         input=pcm, capture_output=True).stderr.decode()
    j = json.loads(log[log.rfind('{'):log.rfind('}') + 1])
    norm = (f"loudnorm=I=-18:TP=-2:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
            f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + inp + ['-af', pre + norm, '-ar', str(SR), '-ac', '1',
                    '-c:a', 'libmp3lame', '-b:a', '96k', path], input=pcm, check=True)

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

FIXES = os.path.join(ROOT, 'scripts', 'review-fixes.json')   # listener's reports: {id: [[line, word, text, kind, note]]}

def apply_reports(key_id, units):
    """Spots the listener reported: ringing / odd voice / held syllables get fresh takes (salted cache key);
    'too fast' / 'too slow' change that phrase's syllable length. Pronunciation is fixed in PRONOUNCE."""
    reps = json.load(open(FIXES, encoding='utf8')).get(key_id, []) if os.path.exists(FIXES) else []
    units, UNIT_SALT_, fac, rang = list(units), {}, {}, {}   # fac[(unit, round)]: within a round the strongest wins, rounds multiply
    for n, wi, word, kind, note, *rnd in reps:
        rnd = rnd[0] if rnd else '|r4'   # each listening round gets its own fresh takes
        cand = [i for i, u in enumerate(units) if u[0] == n and wi in u[2]]
        if units and units[0][5] and word:   # talks: the text may have gained spaces; find the clause by its words
            cand = [i for i, u in enumerate(units) if u[0] == n and word[:6] in u[4].replace(' ', '')] or cand
        if not cand:
            continue
        i = cand[0]; low = note.lower()
        factor = 1.0
        if kind == 'ring':
            rang.setdefault(i, set()).add(rnd)
        if kind == 'tempo' and 'ring' not in low:
            little = any(k in low for k in ('little', 'bit', 'นิด'))
            if 'slow' in low or 'ช้า' in low:
                factor = 0.93 if little else 0.88
            elif 'fast' in low or 'เร็ว' in low:
                factor = 1.08 if little else 1.15
            elif 'short' in low or 'cannot hear' in low or 'สั้น' in low:
                factor = 1.08                       # a syllable swallowed: give the phrase a little more time
        if 'ยืด' in low:
            factor = 0.92                           # drawn-out, stretched voice: read it a little quicker
        if factor == 1.0:
            UNIT_SALT_[i] = max(UNIT_SALT_.get(i, ''), rnd)
        elif abs(factor - 1) > abs(fac.get((i, rnd), 1.0) - 1):
            fac[(i, rnd)] = factor
    for i, rounds in rang.items():   # still ringing after a fresh take: the voice is dragging the phrase, so quicken it
        if len(rounds) >= 2 and (i, 'ring') not in fac:
            fac[(i, 'ring')] = 0.93
    for (i, _), f in fac.items():
        units[i] = units[i][:6] + (units[i][6] * f,)
    return units, UNIT_SALT_

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
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)

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
                units, salts = apply_reports(key_id, units)
                UNIT_SALT.clear(); UNIT_SALT.update(salts)
                global STRICT
                STRICT = None
                if key_id in NATURAL and st['even']:   # one unit per line, natural pace
                    units = [(n, 0, [w for u in units if u[0] == n for w in u[2]], [y for u in units if u[0] == n for y in u[3]],
                              ' '.join(u[4] for u in units if u[0] == n), True, 1.0) for n in range(nlines)]
                    STRICT = (0.11, 0.25, 12.0, [0.05, -0.05, 0.06])
                rates, note = {}, ''
                if sname == 'monk' and lay_segs is not None:   # same voice, same phrases: only pauses/stretch/echo differ
                    segs, note = lay_segs, ' (from normal style)'
                    cues, anchors = assemble(os.path.join(OUT, fname), units, segs, nlines, st)
                    note += f' → stretched to ~{st["syl_ms"] / st["stretch"]:.0f}ms'
                    entry[slot] = fname; entry['cues'][slot] = cues; entry['words'][slot] = anchors
                    print(f'{fname:30s} {nlines} lines · {len(units)} units{note}', flush=True)
                    continue
                best = {}
                want = [sum(u[3]) * st.get('syl_ms', 280) / 1000 * u[6] for u in units]
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
                # a take much shorter than its text needs has dropped a word or a whole sentence: re-take it
                wt = [sum(u[3]) * (u[6] if not u[5] else 1) for u in units]
                for prose in (True, False):
                    ids = [i for i, u in enumerate(units) if u[5] == prose and len(segs[i])]
                    if len(ids) < 3:
                        continue
                    med = float(np.median([len(segs[i]) / SR / wt[i] for i in ids]))
                    short = [i for i in ids if len(segs[i]) / SR / wt[i] < 0.65 * med and sum(units[i][3]) >= 3]
                    if short:
                        for i in short:
                            UNIT_SALT[i] = UNIT_SALT.get(i, '') + '|len'
                        again = render_units(cfg, units, st, rates, short)
                        for i, sg in zip(short, again):
                            if len(sg) > len(segs[i]):
                                segs[i] = sg
                        note += f' · re-took {len(short)} short'
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
