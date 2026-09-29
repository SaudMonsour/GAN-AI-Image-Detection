"""Validate all saved scores, checkpoints, and decompressed source-manifest rows."""
import csv,gzip,io,json,hashlib,tarfile
import numpy as np
import pandas as pd
from PIL import Image
from train import P,CACHE,generator,discriminator,load,norm,scores,jpeg,metrics
def main():
    m=json.loads((P/'metrics.json').read_text()); audit=json.loads((P/'audit.json').read_text()); source=json.loads((P/'data_sources.json').read_text())
    assert hashlib.sha256((P/'train.py').read_bytes()).hexdigest()==audit['training_code_sha256']
    with np.load(CACHE/'subset.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    frame=pd.read_csv(P/'predictions.csv'); assert len(frame)==42000 and not frame.duplicated(['split','model','image_index']).any()
    for (split,name),sub in frame.groupby(['split','model']):
        expected=m['models'][name]['validation'] if split=='validation' else m['models'][name]['test'][split]
        got=metrics(sub.label.to_numpy(),sub.fake_score.to_numpy(dtype=np.float32))
        for key,value in got.items(): assert np.allclose(value,expected[key],atol=1e-8,rtol=0),(split,name,key)
    g=generator(); load(g,'generator.weights.npz'); z=np.random.default_rng(20260928).normal(size=(2000,64)).astype(np.float32)
    fake=np.concatenate([g(z[i:i+128],training=False).numpy() for i in range(0,2000,128)])
    fake=jpeg(np.clip((fake+1)*127.5,0,255).round().astype(np.uint8),95)
    if (P/'generated_test_images.npz').exists():
        with np.load(P/'generated_test_images.npz',allow_pickle=False) as stored: original_fake=stored['images']
    else:
        original_fake=np.load(CACHE/'gan_test_images.npy',allow_pickle=False)
    delta=np.abs(fake.astype(np.int16)-original_fake.astype(np.int16))
    replay=dict(exact_image_matches=int(np.all(delta==0,axis=(1,2,3)).sum()),total_images=2000,max_channel_difference=int(delta.max()),mean_absolute_channel_difference=float(delta.mean()),note='Regeneration is not byte-exact for every image. Original generated test pixels are preserved for exact evaluation; reported training/test scores are unchanged.')
    np.savez_compressed(P/'generated_test_images.npz',images=original_fake)
    with np.load(P/'generated_test_images.npz',allow_pickle=False) as snapshot:
        assert snapshot.files==['images'] and snapshot['images'].shape==(2000,32,32,3)
        assert snapshot['images'].dtype==np.uint8 and np.array_equal(snapshot['images'],original_fake)
    fake=original_fake
    yt=data['y_test']; conditions={'validation':(norm(data['x_validation']),data['y_validation']),'CIFAKE holdout':(norm(data['x_test']),yt),'JPEG quality 50':(norm(jpeg(data['x_test'],50)),yt),'Own GAN holdout':(norm(np.concatenate([data['x_test'][yt==0],fake])),np.r_[np.zeros(2000,dtype=int),np.ones(2000,dtype=int)])}
    reload_errors={}
    weights={'Raw GAN discriminator':'gan_discriminator.weights.npz','CNN from scratch':'scratch_detector.weights.npz','GAN-initialized detector':'gan_initialized_detector.weights.npz'}
    for name,filename in weights.items():
        model=discriminator(); load(model,filename); maximum=0.
        for split,(x,y) in conditions.items():
            sub=frame[(frame.model==name)&(frame.split==split)].sort_values('image_index'); assert np.array_equal(y,sub.label)
            delta=float(np.max(np.abs(scores(model,x)-sub.fake_score.to_numpy(dtype=np.float32)))); maximum=max(maximum,delta)
        assert maximum==0.; reload_errors[name]=maximum
    decoded=gzip.decompress((P/'sample_manifest.csv.gz').read_bytes()); assert hashlib.sha256(decoded).hexdigest()==source['manifest_uncompressed_sha256']
    rows=list(csv.DictReader(io.StringIO(decoded.decode()))); assert len(rows)==18000 and len({r['pixel_sha256'] for r in rows})==18000
    lookup={r['path']:r for r in rows}; count=0
    with tarfile.open(CACHE/'cifake.tar.gz','r:gz') as tar:
        for member in tar:
            relative=member.name.split('/',1)[-1]
            if relative not in lookup: continue
            row=lookup[relative]; raw=tar.extractfile(member).read()
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha']
            px=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),dtype=np.uint8)
            assert hashlib.sha256(px.tobytes()).hexdigest()==row['pixel_sha256']; count+=1
    assert count==18000
    assert hashlib.sha256((CACHE/'cifake.tar.gz').read_bytes()).hexdigest()==source['archive_sha256']
    synthetic_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in fake]
    overlap=set(synthetic_hashes)&{r['pixel_sha256'] for r in rows}; assert not overlap
    finite={}
    for fn in P.glob('*.weights.npz'):
        with np.load(fn,allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            finite[fn.name]={'arrays':len(z.files),'stored_values':sum(z[k].size for k in z.files),'finite':True}
    report=dict(prediction_rows_recalculated=len(frame),full_checkpoint_reload_max_errors=reload_errors,generator_replay=replay,generated_snapshot_full_contents_verified=True,generated_snapshot_pixel_sha256=hashlib.sha256(fake.tobytes()).hexdigest(),unique_generated_test_pixel_hashes=len(set(synthetic_hashes)),generated_overlap_with_source_subset=0,source_manifest_decompressed_rows_checked=count,source_subset_unique_pixel_hashes=18000,source_archive_hash_verified=True,weights=finite)
    (P/'verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
