"""Build the demo page's data (assets/data/examples.json) and audio excerpts.

Inputs are the per-song ensemble CSVs written by tools/run_test_inference.sh
(Mel + PESTO + MIDI, weights 0.5 / 0.1 / 0.4, version-split test set).

    # 1. score all test songs and list candidate excerpt windows
    python tools/build_demo_data.py --pred_dir work --model_repo ../PansoriModeClassification --list

    # 2. write examples.json + mp3 excerpts for the chosen clips (see CLIPS below)
    python tools/build_demo_data.py --pred_dir work --model_repo ../PansoriModeClassification \
        --data_root ../Pansori_Data --out assets

Ground truth comes from <model_repo>/data/Label/label.csv (start/end in ms),
folded to 5 classes exactly as datasets.py does, and sampled at the centre of
each 10 fps frame. Unknown frames are excluded from accuracy.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['Unknown', '우조', '계면조', '아니리', '창조']
CLASS_KEYS = ['unknown', 'ujo', 'gyemyeonjo', 'aniri', 'changjo']
LABEL_MAP = {'Unknown': 0, '경드름': 1, '설렁제': 1, '평조': 1, '우조': 1,
             '계면조': 2, '아니리': 3, '창조': 4}
FPS = 10

# Chosen excerpts: hash -> (start_sec, duration_sec, English title, singer).
# Filled in after reviewing the --list output.
CLIPS = {
    '037e72a6': (170.0, 75.0, 'Chunhyangga — Sipjangga (The Ten Strokes)', 'Kim So-hee'),
    '1c55c3a4': (45.0, 75.0, 'Jeokbyeokga — Jangseung Taryeong (Song of the Guardian Posts)', 'Ahn Sook-sun'),
    '5be7b760': (50.0, 75.0, 'Simcheongga — The Death of Lady Gwak', 'Lee Il-ju'),
    '4597caf5': (110.0, 75.0, 'Jeokbyeokga — Jojo Flees from Jo Ja-ryong', 'Park Dong-jin'),
}


def load_gt(label_csv, hash_key, n_frames):
    df = pd.read_csv(label_csv)
    df = df[df['hash_key'] == hash_key]
    centres_ms = (np.arange(n_frames) + 0.5) * 1000 / FPS
    gt = np.zeros(n_frames, dtype=int)
    for s, e, lab in df[['start', 'end', 'label']].itertuples(index=False):
        gt[(centres_ms >= s) & (centres_ms < e)] = LABEL_MAP[lab]
    return gt


def load_pred(pred_dir, hash_key):
    df = pd.read_csv(Path(pred_dir) / f'{hash_key}_ensemble.csv')
    probs = df[[f'prob_{c}' for c in CLASSES]].to_numpy()
    return df['pred_id'].to_numpy(), probs


def segments(ids):
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            out.append([round(start / FPS, 1), round(i / FPS, 1), CLASS_KEYS[ids[start]]])
            start = i
    return out


def masked_acc(gt, pred):
    m = gt != 0
    return float((gt[m] == pred[m]).mean()) if m.any() else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', required=True)
    p.add_argument('--model_repo', required=True, type=Path)
    p.add_argument('--data_root', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--list', action='store_true', help='print scores and candidate windows')
    p.add_argument('--win', type=float, default=75.0)
    args = p.parse_args()

    label_csv = args.model_repo / 'data/Label/label.csv'
    stems = [s.strip() for s in
             (args.model_repo / 'data/splits/pansori_version_split/test.txt').read_text().splitlines() if s.strip()]

    songs = {}
    all_gt, all_pred = [], []
    for stem in stems:
        h = stem.split('-')[0]
        pred, probs = load_pred(args.pred_dir, h)
        gt = load_gt(label_csv, h, len(pred))
        songs[h] = dict(stem=stem, pred=pred, probs=probs, gt=gt)
        all_gt.append(gt)
        all_pred.append(pred)

    from sklearn.metrics import f1_score
    g, pr = np.concatenate(all_gt), np.concatenate(all_pred)
    m = g != 0
    print(f'Overall masked acc {masked_acc(g, pr):.4f}  macro-F1 '
          f'{f1_score(g[m], pr[m], labels=[1, 2, 3, 4], average="macro"):.4f}  frames {len(g)}')

    if args.list:
        W = int(args.win * FPS)
        for h, s in songs.items():
            gt, pred = s['gt'], s['pred']
            best = None
            for a in range(0, max(1, len(gt) - W), 5 * FPS):
                gw, pw = gt[a:a + W], pred[a:a + W]
                if (gw == 0).mean() > 0.1:
                    continue
                modes = set(gw[gw != 0].tolist())
                n_change = int((np.diff(gw[gw != 0]) != 0).sum())
                if len(modes) < 2:
                    continue
                score = masked_acc(gw, pw) + 0.03 * min(len(modes), 3) + 0.01 * min(n_change, 4)
                if best is None or score > best[0]:
                    best = (score, a / FPS, masked_acc(gw, pw), sorted(CLASSES[c] for c in modes), n_change)
            print(f'{h} song_acc {masked_acc(gt, pred):.3f}  {s["stem"]}')
            if best:
                print(f'    best window @ {best[1]:6.1f}s acc {best[2]:.3f} modes {best[3]} changes {best[4]}')
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'audio').mkdir(exist_ok=True)
    (args.out / 'data').mkdir(exist_ok=True)
    examples = []
    for h, (start, dur, title, singer) in CLIPS.items():
        s = songs[h]
        a, b = int(start * FPS), int((start + dur) * FPS)
        gt, pred, probs = s['gt'][a:b], s['pred'][a:b], s['probs'][a:b]
        clip_id = f'clip_{h}'
        mp3 = args.out / 'audio' / f'{clip_id}.mp3'
        wav = args.data_root / 'Audio_Original' / f'{s["stem"]}.wav'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(start), '-t', str(dur),
                        '-i', str(wav), '-ac', '1', '-b:a', '128k',
                        '-af', f'afade=t=in:d=0.5,afade=t=out:st={dur - 1}:d=1', str(mp3)], check=True)
        examples.append(dict(
            id=clip_id, title=title, singer=singer,
            source_offset_sec=start, duration_sec=dur, fps=FPS,
            accuracy=round(masked_acc(gt, pred), 4),
            gt_segments=segments(gt), pred_segments=segments(pred),
            # Unknown is never predicted in practice; keep the 4 mode columns only.
            probs=[[round(float(x), 3) for x in row[1:]] for row in probs],
            audio=f'assets/audio/{clip_id}.mp3'))
        print(f'{clip_id}: {start}s +{dur}s acc {examples[-1]["accuracy"]:.3f}')
    data = dict(classes=CLASS_KEYS[1:], fps=FPS, examples=examples)
    (args.out / 'data' / 'examples.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
