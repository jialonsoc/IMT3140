import wfdb

record = wfdb.rdheader("1001", pn_dir="ctu-uhb-ctgdb")
# Los metadatos clínicos están en record.comments
for comment in record.comments:
    print(comment)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

record_id = "1001"
fhr = np.load(Path("data/raw") / f"{record_id}_fhr.npy")

fs = 4.0
t = np.arange(len(fhr)) / fs / 60.0

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t, fhr, lw=0.8, color="steelblue")
ax.set_xlabel("Tiempo (min)")
ax.set_ylabel("FHR (bpm)")
ax.set_title(f"FHR registro {record_id}")
ax.set_ylim(50, 210)
ax.grid(alpha=0.3)
plt.show()