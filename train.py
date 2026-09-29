"""Train a DCGAN and test whether its discriminator transfers to CIFAKE detection."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import io,json,time,hashlib,platform
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score,confusion_matrix,log_loss
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
P=Path(__file__).resolve().parent; CACHE=P.parent.parent/'data-cache'
C=dict(seed=42,gan_epochs=10,detector_epochs=6,batch_size=128,latent_dim=64,gan_learning_rate=.0002,detector_learning_rate=.001,gan_beta_1=.5,gradient_clip_norm=1.,decision_threshold=.5,primary_metric='roc_auc',checkpoint_selection='Highest validation ROC-AUC; fixed final epoch for raw GAN',gan_test_latent_seed=20260928,gan_test_images=2000,stress_jpeg_quality=50,gan_test_jpeg_quality=95)
def save(name,obj): (P/name).write_text(json.dumps(obj,indent=2)+'\n')
def generator():
    return tf.keras.Sequential([tf.keras.layers.Input((64,)),tf.keras.layers.Dense(4*4*128,use_bias=False),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU(),tf.keras.layers.Reshape((4,4,128)),tf.keras.layers.Conv2DTranspose(64,4,strides=2,padding='same',use_bias=False),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU(),tf.keras.layers.Conv2DTranspose(32,4,strides=2,padding='same',use_bias=False),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU(),tf.keras.layers.Conv2DTranspose(3,4,strides=2,padding='same',activation='tanh')],name='generator')
def discriminator():
    layers=[tf.keras.layers.Input((32,32,3))]
    for channels in [32,64,128]: layers.extend([tf.keras.layers.Conv2D(channels,4,strides=2,padding='same'),tf.keras.layers.LeakyReLU(negative_slope=.2),tf.keras.layers.Dropout(.25)])
    return tf.keras.Sequential(layers+[tf.keras.layers.Flatten(),tf.keras.layers.Dense(1)],name='detector')
def norm(x): return x.astype(np.float32)/127.5-1
def ds(x,y=None):
    d=tf.data.Dataset.from_tensor_slices(x if y is None else (x,y)); d=d.shuffle(len(x),seed=42,reshuffle_each_iteration=True).batch(128)
    o=tf.data.Options(); o.threading.private_threadpool_size=2; o.experimental_deterministic=True
    return d.with_options(o).prefetch(1)
def scores(model,x):
    return np.concatenate([tf.sigmoid(model(x[i:i+128],training=False)).numpy().reshape(-1) for i in range(0,len(x),128)])
def metrics(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),accuracy=float(accuracy_score(y,pred)),false_positive_rate=float(fp/(tn+fp)),fake_recall=float(tp/(tp+fn)),log_loss=float(log_loss(y,p,labels=[0,1])),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])
def checkpoint(model,name): np.savez(P/name,**{f'w{i}':a for i,a in enumerate(model.get_weights())})
def load(model,name):
    with np.load(P/name,allow_pickle=False) as z: model.set_weights([z[f'w{i}'] for i in range(len(z.files))])
def jpeg(images,quality):
    out=[]
    for a in images:
        f=io.BytesIO(); Image.fromarray(a).save(f,format='JPEG',quality=quality); f.seek(0); out.append(np.asarray(Image.open(f).convert('RGB')))
    return np.stack(out)
def main():
    start=datetime.now(ZoneInfo('Asia/Riyadh')).isoformat(); save('study_config.json',C)
    with np.load(CACHE/'subset.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
    x=norm(a['x_train']); y=a['y_train']; xv=norm(a['x_validation']); yv=a['y_validation']
    tf.keras.utils.set_random_seed(42); g=generator(); d=discriminator()
    go=tf.keras.optimizers.Adam(.0002,beta_1=.5,global_clipnorm=1.); do=tf.keras.optimizers.Adam(.0002,beta_1=.5,global_clipnorm=1.)
    bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    @tf.function(input_signature=[tf.TensorSpec([None,32,32,3],tf.float32),tf.TensorSpec([],tf.int32)])
    def step(real,index):
        z=tf.random.stateless_normal([tf.shape(real)[0],64],seed=tf.stack([42,index]))
        with tf.GradientTape() as gt,tf.GradientTape() as dt:
            fake=g(z,training=True); rl=d(real,training=True); fl=d(fake,training=True)
            dl=.5*(bce(tf.zeros_like(rl),rl)+bce(tf.ones_like(fl),fl)); gl=bce(tf.zeros_like(fl),fl)
        gg=gt.gradient(gl,g.trainable_variables); dg=dt.gradient(dl,d.trainable_variables)
        go.apply_gradients(zip(gg,g.trainable_variables)); do.apply_gradients(zip(dg,d.trainable_variables))
        return gl,dl
    gan_data=ds(x[y==0]); history=[]; index=0; tic=time.perf_counter()
    fixed_z=np.random.default_rng(900).normal(size=(16,64)).astype(np.float32); previews=[]
    for epoch in range(1,11):
        gl=[]; dl=[]
        for real in gan_data:
            gv,dv=step(real,tf.constant(index)); index+=1; gl.append(float(gv)); dl.append(float(dv))
        assert np.isfinite(gl).all() and np.isfinite(dl).all()
        row=dict(phase='GAN',model='DCGAN',epoch=epoch,generator_loss=float(np.mean(gl)),discriminator_loss=float(np.mean(dl)),train_loss=None,validation_auc=None,validation_macro_f1=None)
        history.append(row); print(json.dumps(row),flush=True)
        previews.append(np.clip((g(fixed_z,training=False).numpy()+1)*127.5,0,255).round().astype(np.uint8))
        pd.DataFrame(history).to_csv(P/'training_history.csv',index=False)
    gan_seconds=time.perf_counter()-tic; np.save(CACHE/'gan_progress.npy',np.stack(previews)); checkpoint(g,'generator.weights.npz'); checkpoint(d,'gan_discriminator.weights.npz')
    pv=scores(d,xv); results={'Raw GAN discriminator':dict(parameters=d.count_params(),selected_epoch=10,validation=metrics(yv,pv))}; models={'Raw GAN discriminator':d}; val_scores={'Raw GAN discriminator':pv}
    for name,file,pretrained in [('CNN from scratch','scratch_detector.weights.npz',False),('GAN-initialized detector','gan_initialized_detector.weights.npz',True)]:
        tf.keras.utils.set_random_seed(42); model=discriminator()
        if pretrained: load(model,'gan_discriminator.weights.npz')
        model.compile(optimizer=tf.keras.optimizers.Adam(.001,global_clipnorm=1.),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        train=ds(x,y.astype(np.float32).reshape(-1,1)); best=-1; best_epoch=0; tic=time.perf_counter()
        for epoch in range(1,7):
            h=model.fit(train,epochs=1,verbose=0,shuffle=False); p=scores(model,xv); vm=metrics(yv,p)
            row=dict(phase='Supervised',model=name,epoch=epoch,generator_loss=None,discriminator_loss=None,train_loss=float(h.history['loss'][0]),validation_auc=vm['roc_auc'],validation_macro_f1=vm['macro_f1'])
            history.append(row); print(json.dumps(row),flush=True)
            if vm['roc_auc']>best: best=vm['roc_auc']; best_epoch=epoch; checkpoint(model,file)
            pd.DataFrame(history).to_csv(P/'training_history.csv',index=False)
        seconds=time.perf_counter()-tic; load(model,file); p=scores(model,xv); val_scores[name]=p
        models[name]=model; results[name]=dict(parameters=model.count_params(),selected_epoch=best_epoch,supervised_seconds=seconds,validation=metrics(yv,p))
    chosen=max(results,key=lambda n:results[n]['validation']['roc_auc'])
    save('selection.json',dict(selected_model=chosen,selection_metric='validation ROC-AUC',test_seen=False,threshold=.5,models=results))
    rows=[]
    for name,p in val_scores.items():
        for i,(yi,pi) in enumerate(zip(yv,p)): rows.append(dict(split='validation',model=name,image_index=i,label=int(yi),fake_score=float(pi)))
    # No model/threshold choice after this point. Three predefined evaluation conditions.
    xt=norm(a['x_test']); yt=a['y_test']; rng=np.random.default_rng(20260928); z=rng.normal(size=(2000,64)).astype(np.float32)
    generated=np.concatenate([g(z[i:i+128],training=False).numpy() for i in range(0,2000,128)])
    generated=np.clip((generated+1)*127.5,0,255).round().astype(np.uint8); generated=jpeg(generated,95); np.save(CACHE/'gan_test_images.npy',generated)
    conditions={'CIFAKE holdout':(xt,yt),'JPEG quality 50':(norm(jpeg(a['x_test'],50)),yt),'Own GAN holdout':(norm(np.concatenate([a['x_test'][yt==0],generated])),np.r_[np.zeros(2000,dtype=int),np.ones(2000,dtype=int)])}
    verification={}
    for name,model in models.items():
        results[name]['test']={}
        for condition,(xx,yy) in conditions.items():
            p=scores(model,xx); results[name]['test'][condition]=metrics(yy,p)
            for i,(yi,pi) in enumerate(zip(yy,p)): rows.append(dict(split=condition,model=name,image_index=i,label=int(yi),fake_score=float(pi)))
        file={'Raw GAN discriminator':'gan_discriminator.weights.npz','CNN from scratch':'scratch_detector.weights.npz','GAN-initialized detector':'gan_initialized_detector.weights.npz'}[name]
        clone=discriminator(); load(clone,file); err=float(np.max(np.abs(scores(model,xt[:128])-scores(clone,xt[:128])))); assert err==0
        verification[name]={'checkpoint_reload_max_score_error':err}
    baseline={}
    for condition,(xx,yy) in conditions.items():
        p=np.zeros(len(yy),dtype=np.float32); baseline[condition]=metrics(yy,p)
    results['Always-real baseline']={'validation':metrics(yv,np.zeros(len(yv),dtype=np.float32)),'test':baseline}
    pd.DataFrame(rows).to_csv(P/'predictions.csv',index=False)
    save('metrics.json',dict(selected_model=chosen,positive_class='AI generated',primary_condition='CIFAKE holdout',gan_training_seconds=gan_seconds,models=results)); save('verification.json',verification)
    save('audit.json',dict(started_at=start,finished_at=datetime.now(ZoneInfo('Asia/Riyadh')).isoformat(),python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,device='CPU',deterministic_ops=True,training_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config_sha256=hashlib.sha256((P/'study_config.json').read_bytes()).hexdigest(),gan_optimizer_steps=index,generator_parameters=g.count_params()))
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__': main()
