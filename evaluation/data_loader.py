import pandas as pd, requests, io, gzip, re
from sklearn.utils import shuffle

URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/"
       "matrix/GSE2034_series_matrix.txt.gz")
blob = requests.get(URL, timeout=120); blob.raise_for_status()

with gzip.open(io.BytesIO(blob.content), mode="rt", encoding="utf-8") as fh:
    lines = fh.readlines()

# ── sample IDs ---------------------------------------------------
acc_row   = next(l for l in lines if l.startswith('!Sample_geo_accession'))
sample_id = [c.strip().strip('"') for c in acc_row.split('\t')[1:]]

# ── relapse row --------------------------------------------------
rel_row   = next(l for l in lines if 'bone relapses' in l.lower())
raw_vals  = rel_row.split('\t')[1:]        # keep quotes for regex

# pull the last digit (0 or 1) right before the end quote / line end
get_bin = lambda s: int(re.search(r'([01])\s*"?$', s).group(1))
labels  = {sid: get_bin(txt) for sid, txt in zip(sample_id, raw_vals)}
y_full  = pd.Series(labels, name="relapse")

print("label counts\n", y_full.value_counts())      # sanity-check

# ── expression matrix -------------------------------------------
beg = lines.index('!series_matrix_table_begin\n') + 1
end = lines.index('!series_matrix_table_end\n')
expr = pd.read_csv(io.StringIO(''.join(lines[beg:end])),
                   sep='\t', index_col=0)

X_full = expr.T.loc[y_full.index]                   # (286 × 22 283)
print("X_full shape:", X_full.shape)

# ── 15 positives + 15 negatives -------------------------------
pos = y_full[y_full == 1].sample(15, random_state=42).index
neg = y_full[y_full == 0].sample(15, random_state=42).index
sel = pos.union(neg)

X_small, y_small = shuffle(X_full.loc[sel], y_full.loc[sel],
                           random_state=42)

print("mini-set:", X_small.shape)
print(y_small.value_counts())
