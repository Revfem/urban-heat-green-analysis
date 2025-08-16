
import argparse, numpy as np, rasterio, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--cmap", default="inferno")     # ex: inferno, RdBu_r, magma, viridis
ap.add_argument("--sym", action="store_true")    # scale simetric față de 0 (util pt. SUHI)
ap.add_argument("--label", default="")           # eticheta barei de culori
ap.add_argument("--pclip", type=float, default=2)# percent clip (2..98)
args = ap.parse_args()

with rasterio.open(args.input) as src:
    a = src.read(1).astype(float)

finite = np.isfinite(a)
if not finite.any():
    raise SystemExit("No finite pixels!")

p_lo, p_hi = np.nanpercentile(a[finite], (args.pclip, 100-args.pclip))
if args.sym:
    m = max(abs(p_lo), abs(p_hi))
    vmin, vmax = -m, m
else:
    vmin, vmax = p_lo, p_hi

plt.figure(figsize=(8,6))
im = plt.imshow(a, vmin=vmin, vmax=vmax, cmap=args.cmap)
cb = plt.colorbar(im)
if args.label: cb.set_label(args.label)
plt.axis("off")
plt.tight_layout()
plt.savefig(args.output, dpi=200)
