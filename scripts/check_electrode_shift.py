"""
Diagnose why some Hyser subjects fail across sessions: signal quality and electrode shift.

For each subject it (optionally) downloads the dynamic raw signals of both sessions, then computes
from the raw EMG (10 to 500 Hz bandpass + 50 Hz notch, first 0.25 s of each segment dropped):

  1. channel quality per session: mean channel RMS, count of flat channels (RMS < 0.1 x median)
     and noisy channels (RMS > 5 x median), and how many of them are the same channels in both sessions
  2. overall amplitude change: median channel RMS of session 2 / session 1
  3. spatial pattern stability: for each gesture, the correlation of the 256-channel RMS map
       - between two halves of the SAME session (split by trial parity)  -> within-session reference
       - between session 1 and session 2                                  -> cross-session
     A low cross-session value compared with the within-session value means the pattern moved.
  4. nearest-centroid gesture classifier on the RMS maps (centroids from session 1 odd trials,
     tested on session 1 even trials and on session 2): a model-free check of the accuracy drop
  5. electrode shift estimate: for each 8 x 8 array, the integer shift (rows, columns, +-2 electrodes,
     10 mm spacing) that maximises the correlation of the gesture-average RMS map between sessions.
     ASSUMES the 64 channels of an array are laid out row by row on the 8 x 8 grid. Treat as indicative.

Usage (from the repo root, with the project environment active):
    python scripts/check_electrode_shift.py --subjects 02,20,06,19 --raw-root ~/hyser_data/hyser
Add --skip-download if the raw files are already there; add --delete-raw to remove them afterwards.
Results go to reports/electrode_shift/.
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
S3_BASE = "s3://physionet-open/hd-semg/2.0.0/pr_dataset/"
FS = 2048
SKIP = int(0.25 * FS)
BP = dict(low_hz=10, high_hz=500, order=4)
NOTCH = dict(freq_hz=50, quality_factor=30)


def chain(x):
    from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
    sos = butter(BP["order"], [BP["low_hz"] / (FS / 2), BP["high_hz"] / (FS / 2)], btype="bandpass", output="sos")
    b, a = iirnotch(NOTCH["freq_hz"], NOTCH["quality_factor"], FS)
    return sosfiltfilt(tf2sos(b, a), sosfiltfilt(sos, x, axis=-1), axis=-1)


def trial_rms(dat):
    import wfdb
    rec = wfdb.rdrecord(str(Path(dat).with_suffix("")))
    x = rec.p_signal.T[:, SKIP:]
    x = chain(x - x.mean(axis=1, keepdims=True))
    return np.sqrt((x ** 2).mean(axis=1))          # (256,)


def load_session(raw_root, sub, sess):
    d = Path(raw_root) / f"subject{sub}_session{sess}"
    labels = np.loadtxt(d / "label_dynamic.txt", delimiter=",", dtype=int).ravel()
    X, y, k = [], [], []
    for i, g in enumerate(labels, start=1):
        f = d / f"dynamic_raw_sample{i}.dat"
        if f.exists():
            X.append(trial_rms(f)); y.append(int(g)); k.append(i)
    return np.array(X), np.array(y), np.array(k)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gesture_maps(X, y, parity=None, trial_idx=None):
    maps = {}
    for g in np.unique(y):
        sel = np.where(y == g)[0]
        if parity is not None:
            sel = sel[np.arange(len(sel)) % 2 == parity]
        if len(sel):
            maps[g] = X[sel].mean(axis=0)
    return maps


def norm_map(v):
    v = np.log(v + 1e-9)            # log RMS: amplitude ratios become additive
    return v - v.mean()


def best_shift(m1, m2, maxs=2):
    """Integer (dr, dc) shift of m2 relative to m1 maximising correlation on the overlap, 8x8 grid."""
    a = m1.reshape(8, 8); b = m2.reshape(8, 8); best = (0, 0, -9, None)
    zero = None
    for dr in range(-maxs, maxs + 1):
        for dc in range(-maxs, maxs + 1):
            ar = a[max(0, dr):8 + min(0, dr), max(0, dc):8 + min(0, dc)]
            br = b[max(0, -dr):8 + min(0, -dr), max(0, -dc):8 + min(0, -dc)]
            c = corr(ar.ravel(), br.ravel())
            if dr == 0 and dc == 0: zero = c
            if c > best[2]: best = (dr, dc, c, None)
    return best[0], best[1], best[2], zero


def analyse(X1, y1, X2, y2):
    out = {}
    med = lambda X: float(np.median(X.mean(axis=0)))
    ch1, ch2 = X1.mean(axis=0), X2.mean(axis=0)
    flat1, flat2 = ch1 < 0.1 * np.median(ch1), ch2 < 0.1 * np.median(ch2)
    nois1, nois2 = ch1 > 5 * np.median(ch1), ch2 > 5 * np.median(ch2)
    out.update(median_rms_s1=med(X1), median_rms_s2=med(X2), amp_ratio_s2_over_s1=med(X2) / med(X1),
               flat_s1=int(flat1.sum()), flat_s2=int(flat2.sum()), flat_both=int((flat1 & flat2).sum()),
               noisy_s1=int(nois1.sum()), noisy_s2=int(nois2.sum()), noisy_both=int((nois1 & nois2).sum()))
    # pattern stability
    w1 = [(gesture_maps(X1, y1, 0), gesture_maps(X1, y1, 1)), (gesture_maps(X2, y2, 0), gesture_maps(X2, y2, 1))]
    c1 = gesture_maps(X1, y1); c2 = gesture_maps(X2, y2)
    gs = sorted(set(c1) & set(c2))
    r_w1 = [corr(norm_map(w1[0][0][g]), norm_map(w1[0][1][g])) for g in gs if g in w1[0][0] and g in w1[0][1]]
    r_w2 = [corr(norm_map(w1[1][0][g]), norm_map(w1[1][1][g])) for g in gs if g in w1[1][0] and g in w1[1][1]]
    r_x = [corr(norm_map(c1[g]), norm_map(c2[g])) for g in gs]
    out.update(r_within_s1=float(np.mean(r_w1)), r_within_s2=float(np.mean(r_w2)), r_cross=float(np.mean(r_x)))
    out["per_gesture_r_cross"] = {int(g): r for g, r in zip(gs, r_x)}
    # nearest centroid on log RMS maps
    cen = gesture_maps(X1, y1, 0); G = sorted(cen); C = np.stack([norm_map(cen[g]) for g in G])
    def acc(X, y, mask=None):
        idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
        ok = [G[int(np.argmax([corr(norm_map(X[i]), c) for c in C]))] == y[i] for i in idx]
        return float(np.mean(ok)) if ok else float("nan")
    par = np.zeros(len(y1), bool)
    for g in np.unique(y1):
        sel = np.where(y1 == g)[0]; par[sel[np.arange(len(sel)) % 2 == 1]] = True
    out.update(centroid_acc_same=acc(X1, y1, par), centroid_acc_cross=acc(X2, y2))
    # shift per array, gesture-average maps
    shifts = []
    for a in range(4):
        sl = slice(64 * a, 64 * (a + 1)); res = []
        for g in gs:
            res.append(best_shift(norm_map(c1[g])[sl] if False else np.log(c1[g][sl] + 1e-9), np.log(c2[g][sl] + 1e-9)))
        drs = [r[0] for r in res]; dcs = [r[1] for r in res]
        shifts.append(dict(array=a + 1, median_dr=float(np.median(drs)), median_dc=float(np.median(dcs)),
                           frac_nonzero=float(np.mean([(r[0], r[1]) != (0, 0) for r in res])),
                           mean_gain=float(np.mean([r[2] - r[3] for r in res]))))
    out["shift"] = shifts
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="02,20,06,19")
    ap.add_argument("--raw-root", type=Path, default=Path("~/hyser_data/hyser"))
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "electrode_shift")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--delete-raw", action="store_true")
    a = ap.parse_args()
    raw = a.raw_root.expanduser().resolve(); a.out.mkdir(parents=True, exist_ok=True)
    subs = [f"{int(s):02d}" for s in a.subjects.split(",")]
    for sub in subs:
        if not a.skip_download:
            for sess in (1, 2):
                folder = f"subject{sub}_session{sess}"
                cmd = ["aws", "s3", "sync", "--no-sign-request", "--only-show-errors", f"{S3_BASE}{folder}/", str(raw / folder),
                       "--exclude", "*", "--include", "dynamic_raw_sample*", "--include", "label_dynamic.txt"]
                for attempt in range(1, 9):           # dropped connections are common; sync resumes where it stopped
                    if subprocess.run(cmd).returncode == 0:
                        break
                    print(f"  download of {folder} incomplete (attempt {attempt}/8), retrying in 15 s", flush=True)
                    time.sleep(15)
                else:
                    sys.exit(f"download of {folder} kept failing; check the connection and run again")
        print(f"subject{sub}: reading", flush=True)
        X1, y1, _ = load_session(raw, sub, 1); X2, y2, _ = load_session(raw, sub, 2)
        res = analyse(X1, y1, X2, y2); res.update(subject=f"S{sub}", n_trials_s1=len(y1), n_trials_s2=len(y2))
        np.savez(a.out / f"rms_S{sub}.npz", X1=X1, y1=y1, X2=X2, y2=y2)
        json.dump(res, open(a.out / f"S{sub}.json", "w"), indent=1)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("per_gesture_r_cross", "shift")}, flush=True)
        if a.delete_raw:
            for sess in (1, 2): shutil.rmtree(raw / f"subject{sub}_session{sess}", ignore_errors=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
