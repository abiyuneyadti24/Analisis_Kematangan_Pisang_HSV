"""Analisis kematangan pisang dengan segmentasi warna HSV (OpenCV).
Pakai: python analisis_pisang.py <folder_output> <foto1> [foto2 ...]
"""
import sys, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
files = sys.argv[2:]
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- 1-2. Tiap foto: pisahkan objek dari latar (latar terang, saturasi rendah),
#          lalu ambil tiap tandan/buah (komponen besar), urut kiri ke kanan ---
objs = []   # (crop BGR, mask objek)
for f in files:
    img = cv2.imread(f)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    fg = ((S > 70) | (V < 110)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    amax = stats[1:, cv2.CC_STAT_AREA].max()
    idx = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.3 * amax]
    for i in sorted(idx, key=lambda i: stats[i, cv2.CC_STAT_LEFT]):
        x, y, w, h, _a = stats[i]
        objs.append((img[y:y+h, x:x+w].copy(), lab[y:y+h, x:x+w] == i))
N = len(objs)

# --- 3. Rentang HSV (OpenCV: H 0-179, S/V 0-255) ---
def masks(h, s, v):
    hijau  = (h >= 27) & (h <= 85) & (s >= 60) & (v >= 60)
    kuning = (h >= 18) & (h <= 26) & (s >= 90) & (v >= 140)
    # cokelat: rona jingga/kuning yang gelap, atau sangat gelap (hitam-cokelat)
    cokelat = (((h <= 35) | (h >= 170)) & (s >= 40) & (v < 140)) | (v < 60)
    cokelat &= ~kuning & ~hijau
    return hijau, kuning, cokelat

rows = []
fig, ax = plt.subplots(N, 5, figsize=(15, 3 * N))
for r, (crop, obj) in enumerate(objs):
    nm = f"Citra {r+1}"
    hsvc = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop[~obj] = 255
    cv2.imwrite(f"{out}/citra_{r+1}.png", crop)
    ch, cs, cv_ = cv2.split(hsvc)
    g, yl, br = [m & obj for m in masks(ch, cs, cv_)]
    tot = obj.sum()
    p = [100 * m.sum() / tot for m in (g, yl, br)]
    lain = 100 - sum(p)
    kelas = ["Mentah", "Matang", "Terlalu matang/busuk"][int(np.argmax(p))]
    if kelas == "Matang" and p[2] >= 5: kelas = "Matang lanjut (mulai berbintik)"
    rows.append((nm, int(tot), *p, lain, kelas))
    ax[r, 0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax[r, 0].set_title(nm)
    ax[r, 1].imshow(ch, cmap="hsv", vmin=0, vmax=179); ax[r, 1].set_title("Kanal Hue")
    for c, (m, t) in enumerate(zip((g, yl, br), ("Hijau", "Kuning", "Cokelat"))):
        ax[r, 2+c].imshow(m, cmap="gray"); ax[r, 2+c].set_title(f"Mask {t}: {p[c]:.1f}%")
    for a in ax[r]: a.axis("off")
plt.tight_layout(); plt.savefig(f"{out}/segmentasi_hsv.png", dpi=130); plt.close()

print(f"{'Citra':18s}{'Piksel':>8s}{'Hijau%':>9s}{'Kuning%':>9s}{'Cokelat%':>10s}{'Lain%':>8s}  Klasifikasi")
for r in rows: print(f"{r[0]:18s}{r[1]:8d}{r[2]:9.2f}{r[3]:9.2f}{r[4]:10.2f}{r[5]:8.2f}  {r[6]}")

# --- 4. Grafik perbandingan ---
xs = np.arange(N); wd = 0.26
fig, a = plt.subplots(figsize=(10.5, 5.2))
for j, (t, col) in enumerate((("Hijau", "#4c9a2a"), ("Kuning", "#f2c200"), ("Cokelat", "#6b3e1e"))):
    vals = [r[2+j] for r in rows]
    b = a.bar(xs + (j-1)*wd, vals, wd, label=t, color=col, edgecolor="black", linewidth=.6)
    a.bar_label(b, fmt="%.1f%%", padding=2, fontsize=9)
a.set_xticks(xs); a.set_xticklabels([f"{r[0]}\n{r[6]}" for r in rows], fontsize=9)
a.set_ylabel("Persentase piksel buah (%)"); a.set_ylim(0, 110)
a.set_title("Perbandingan persentase warna hijau, kuning, dan cokelat")
a.legend(); a.grid(axis="y", alpha=.3); a.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f"{out}/grafik_perbandingan.png", dpi=150)
