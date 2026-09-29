"""Fetch an immutable CIFAKE release and build an audited, balanced subset."""
from pathlib import Path
import io, json, hashlib, tarfile, urllib.request, gzip, csv
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parent
CACHE=P.parent.parent/'data-cache'
COMMIT='e112a942abaecd02b6b1f6f646c807d56be8fb62'
URL=f'https://codeload.github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/tar.gz/{COMMIT}'
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    CACHE.mkdir(exist_ok=True); archive=CACHE/'cifake.tar.gz'
    if not archive.exists():
        print('Downloading pinned author-provided CIFAKE archive',flush=True)
        with urllib.request.urlopen(URL,timeout=60) as r, archive.with_suffix('.part').open('wb') as f:
            while b:=r.read(1024*1024): f.write(b)
        archive.with_suffix('.part').rename(archive)
    blobs={}; source_readme=''
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            if not m.isfile(): continue
            rel=m.name.split('/',1)[-1]
            if rel=='README.md': source_readme=tar.extractfile(m).read().decode()
            elif rel.endswith('.jpg') and rel.startswith('DATASET/'):
                blobs[rel]=tar.extractfile(m).read()
    counts={f'{split}/{label}':sum(p.startswith(f'DATASET/{split}/{label}/') for p in blobs) for split in ['train','test'] for label in ['REAL','FAKE']}
    assert counts=={'train/REAL':50000,'train/FAKE':50000,'test/REAL':10000,'test/FAKE':10000},counts
    seen={}; records=[]; duplicates=[]; xs={k:[] for k in ['train','validation','test']}; ys={k:[] for k in xs}
    for original,limit in [('train',7000),('test',2000)]:
        for label,cls in enumerate(['REAL','FAKE']):
            paths=sorted((p for p in blobs if p.startswith(f'DATASET/{original}/{cls}/')),key=lambda p:sha(('42:'+p).encode()))
            kept=0
            for path in paths:
                image=Image.open(io.BytesIO(blobs[path])); assert image.size==(32,32)
                pixels=np.asarray(image.convert('RGB'),dtype=np.uint8); ph=sha(pixels.tobytes())
                if ph in seen:
                    duplicates.append(dict(path=path,duplicate_of=seen[ph]['path'],label=label,previous_label=seen[ph]['label'])); continue
                split='test' if original=='test' else ('train' if kept<6000 else 'validation')
                raw=blobs[path]; gh=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                row=dict(path=path,split=split,label=label,git_blob_sha=gh,pixel_sha256=ph)
                seen[ph]=row; records.append(row); xs[split].append(pixels); ys[split].append(label); kept+=1
                if kept==limit: break
            assert kept==limit
    arrays={}
    for split in xs:
        arrays[f'x_{split}']=np.stack(xs[split]); arrays[f'y_{split}']=np.array(ys[split],dtype=np.int32)
    np.savez(CACHE/'subset.npz',**arrays)
    text=io.StringIO(); writer=csv.DictWriter(text,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    encoded=text.getvalue().encode(); (P/'sample_manifest.csv.gz').write_bytes(gzip.compress(encoded,mtime=0))
    # Inspect every decompressed metadata row and compare it with the pinned source.
    decoded=gzip.decompress((P/'sample_manifest.csv.gz').read_bytes()); assert decoded==encoded
    for row in csv.DictReader(io.StringIO(decoded.decode())):
        raw=blobs[row['path']]; assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha']
        px=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),dtype=np.uint8); assert sha(px.tobytes())==row['pixel_sha256']
    (P/'duplicates_removed.json').write_text(json.dumps(duplicates,indent=2)+'\n')
    (P/'DATA_LICENSE.txt').write_text(source_readme.split('## License',1)[1].strip()+'\n')
    quality=dict(source_url=URL,source_repository='jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images',source_commit=COMMIT,archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,source_counts=counts,subset_counts={k:{'real':int(np.sum(arrays[f'y_{k}']==0)),'fake':int(np.sum(arrays[f'y_{k}']==1))} for k in xs},decoded_shape=[32,32,3],selection='Sort SHA-256 of 42:path within source split/class; retain first unique pixels; 6000 training + 1000 validation per training class; 2000 test per test class',exact_duplicate_candidates_removed=len(duplicates),cross_label_duplicate_candidates=sum(r['label']!=r['previous_label'] for r in duplicates),unique_pixel_hashes=len(seen),manifest_rows_verified_against_pinned_archive=len(records),manifest_uncompressed_sha256=sha(encoded),dataset_snapshot_stored=False)
    (P/'data_sources.json').write_text(json.dumps(quality,indent=2)+'\n'); print(json.dumps(quality,indent=2),flush=True)
if __name__=='__main__': main()
