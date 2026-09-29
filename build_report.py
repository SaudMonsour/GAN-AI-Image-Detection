"""Create figures and documentation from recorded predictions, never example scores."""
from pathlib import Path
import json,gzip,io,html,contextlib,os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; CACHE=P.parent.parent/'data-cache'
def read(name): return json.loads((P/name).read_text())
def save(name,obj): (P/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
    m=read('metrics.json'); models=m['models']; chosen=m['selected_model']; source=read('data_sources.json')
    frame=pd.read_csv(P/'predictions.csv'); hist=pd.read_csv(P/'training_history.csv'); manifest=pd.read_csv(P/'sample_manifest.csv.gz')
    with np.load(CACHE/'subset.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    names=list(models)[:3]; short=['Raw GAN','Scratch CNN','GAN initialized']; colors=['#64748b','#2563eb','#ea7b24']; conditions=['CIFAKE holdout','JPEG quality 50','Own GAN holdout']
    primary=frame[frame.split=='CIFAKE holdout']; y=primary[primary.model==names[0]].sort_values('image_index').label.to_numpy()
    ps={n:primary[primary.model==n].sort_values('image_index').fake_score.to_numpy() for n in names}
    rng=np.random.default_rng(4242); boot={n:[] for n in names}; ix=[np.flatnonzero(y==i) for i in [0,1]]
    for _ in range(500):
        b=np.r_[rng.choice(ix[0],len(ix[0]),replace=True),rng.choice(ix[1],len(ix[1]),replace=True)]
        for n in names: boot[n].append(roc_auc_score(y[b],ps[n][b]))
    uncertainty={n:{'roc_auc_95_percentile_interval':np.percentile(v,[2.5,97.5]).tolist()} for n,v in boot.items()}
    uncertainty['gan_initialized_minus_scratch']={'point':models[names[2]]['test'][conditions[0]]['roc_auc']-models[names[1]]['test'][conditions[0]]['roc_auc'],'paired_95_percentile_interval':np.percentile(np.array(boot[names[2]])-np.array(boot[names[1]]),[2.5,97.5]).tolist()}
    uncertainty['method']='500 paired, class-stratified bootstrap resamples; seed 4242; conditional on these fixed trained checkpoints; post-training uncertainty analysis'
    save('uncertainty.json',uncertainty)
    selected=primary[primary.model==chosen].sort_values('image_index').copy(); selected['predicted_label']=(selected.fake_score>=.5).astype(int)
    selected['path']=manifest[manifest.split=='test'].path.to_numpy(); mistakes=selected[selected.label!=selected.predicted_label]
    mistakes.to_csv(P/'error_analysis.csv',index=False)
    figs=P/'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
    def figure(name,**layout): plt.tight_layout(**layout); plt.savefig(figs/name,dpi=150,bbox_inches='tight'); plt.close()
    fig,axes=plt.subplots(4,8,figsize=(10,5.8))
    for row in range(4):
        label=row//2; ids=np.flatnonzero(data['y_train']==label)[(row%2)*8:(row%2+1)*8]
        for col,i in enumerate(ids):
            axes[row,col].imshow(data['x_train'][i],interpolation='nearest'); axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if col==0: axes[row,col].set_ylabel('Real' if label==0 else 'Stable\nDiffusion',fontsize=10)
    fig.suptitle('Fixed first examples from the selected training subset'); figure('dataset-examples.png')
    progress=np.load(CACHE/'gan_progress.npy',allow_pickle=False); fig,axes=plt.subplots(4,8,figsize=(10,5.8))
    for row,epoch in enumerate([1,4,7,10]):
        for col in range(8):
            axes[row,col].imshow(progress[epoch-1,col],interpolation='nearest'); axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if col==0: axes[row,col].set_ylabel(f'Epoch {epoch}')
    fig.suptitle('GAN outputs at fixed latent inputs — no sample selection'); figure('gan-progress.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); gh=hist[hist.phase=='GAN']
    axes[0].plot(gh.epoch,gh.generator_loss,label='Generator'); axes[0].plot(gh.epoch,gh.discriminator_loss,label='Discriminator'); axes[0].set(title='Adversarial training',xlabel='Epoch',ylabel='Training loss'); axes[0].legend()
    for n,c in zip(names[1:],colors[1:]):
        h=hist[hist.model==n]; axes[1].plot(h.epoch,h.validation_auc,'o-',label=n,color=c)
    axes[1].set(title='Supervised detector validation',xlabel='Epoch',ylabel='Validation ROC-AUC'); axes[1].legend(fontsize=8); figure('learning-curves.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,key,title in zip(axes,['roc_auc','macro_f1'],['ROC-AUC','Macro-F1 at threshold 0.5']):
        vals=[models[n]['test'][conditions[0]][key] for n in models]; b=ax.bar(short+['Always real'],vals,color=colors+['#cbd5e1']); ax.bar_label(b,fmt='%.3f',padding=3,fontsize=9); ax.set_ylim(0,1.09); ax.set_title(title); ax.tick_params(axis='x',labelrotation=15)
    figure('model-comparison.png')
    plt.figure(figsize=(6.7,5))
    for n,c in zip(names,colors):
        fpr,tpr,_=roc_curve(y,ps[n]); plt.plot(fpr,tpr,color=c,label=f"{n}: {models[n]['test'][conditions[0]]['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'--',color='#94a3b8'); plt.xlabel('False-positive rate'); plt.ylabel('True-positive rate'); plt.title('CIFAKE test ROC curves'); plt.legend(fontsize=8); figure('roc-curve.png')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,n,label in zip(axes,names,short):
        cm=np.array(models[n]['test'][conditions[0]]['confusion_matrix']); ax.imshow(cm,cmap='Blues',vmin=0,vmax=2000)
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>1000 else '#0f172a',fontsize=14)
        ax.set(title=label,xticks=[0,1],yticks=[0,1],xticklabels=['Real','AI'],yticklabels=['Real','AI'],xlabel='Predicted',ylabel='Actual')
    figure('confusion-matrix.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5)); positions=np.arange(3)
    for ax,key,title in zip(axes,['roc_auc','false_positive_rate'],['ROC-AUC across conditions','Real images falsely flagged as AI']):
        for j,(n,c,label) in enumerate(zip(names,colors,short)):
            vals=[models[n]['test'][condition][key] for condition in conditions]; ax.bar(positions+(j-1)*.24,vals,width=.24,color=c,label=label)
        ax.set_xticks(positions,['CIFAKE','JPEG Q50','Own GAN']); ax.set_ylim(0,1.06); ax.set_title(title); ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3)
    figure('robustness.png')
    fp=mistakes[mistakes.label==0].nlargest(4,'fake_score'); fn=mistakes[mistakes.label==1].nsmallest(4,'fake_score'); fig,axes=plt.subplots(2,4,figsize=(9,6.8))
    for row,subset in enumerate([fp,fn]):
        for ax,(_,r) in zip(axes[row],subset.iterrows()):
            ax.imshow(data['x_test'][int(r.image_index)],interpolation='nearest'); ax.axis('off'); ax.set_title(f"Actual: {'real' if r.label==0 else 'AI'}\nAI score: {r.fake_score:.3f}",fontsize=10)
    fig.suptitle('Selected detector: four most confident errors per class'); figure('error-examples.png',h_pad=4)
    own=models[names[0]]['test']['Own GAN holdout']; raw=models[names[0]]['test'][conditions[0]]; best=models[chosen]['test'][conditions[0]]; stress=models[chosen]['test'][conditions[1]]
    valrows='\n'.join(f"| {n} | {r['validation']['roc_auc']:.4f} | {r.get('selected_epoch','N/A')} |" for n,r in models.items())
    testrows='\n'.join(f"| {n} | {r['test'][conditions[0]]['roc_auc']:.4f} | {r['test'][conditions[0]]['macro_f1']:.4f} | {100*r['test'][conditions[0]]['accuracy']:.2f}% |" for n,r in models.items())
    robustness='\n'.join(f"| {n} | {r['test'][conditions[0]]['roc_auc']:.4f} | {r['test'][conditions[1]]['roc_auc']:.4f} | {r['test'][conditions[2]]['roc_auc']:.4f} |" for n,r in models.items())
    ci=uncertainty[chosen]['roc_auc_95_percentile_interval']; delta=uncertainty['gan_initialized_minus_scratch']
    comparison='did not improve' if delta['point']<=0 else 'improved'
    readme=f'''# Can a GAN Discriminator Detect AI-Generated Images?

Train a TensorFlow GAN and test whether its discriminator transfers from its own generated images to Stable Diffusion image detection.

---

## Overview

* **Task:** Binary classification of real versus AI-generated images.
* **Dataset:** 18,000 CIFAKE images; 32 × 32 RGB pixels (3,072 pixel-channel values per image).
* **Selected Model:** **{chosen}** (test **ROC-AUC = {best['roc_auc']:.4f}**), selected by validation ROC-AUC.
* **Baseline Comparison:** Always-real baseline achieved ROC-AUC = 0.5000 and accuracy = 50.00% on the balanced test set.

**Research question:** Does adversarial pretraining improve a detector of images from a different generation method?

The raw discriminator achieved **{own['roc_auc']:.6f} ROC-AUC on fresh outputs from its own GAN**, but only **{raw['roc_auc']:.4f} on Stable Diffusion images**. GAN initialization {comparison} the supervised detector's primary test ROC-AUC in this fixed-budget, one-seed experiment. Recognizing one weak generator's artifacts is insufficient evidence of general AI-image detection.

---

## Key Results

### Training Validation

One fixed validation partition was used; cross-validation and fold SD are not applicable. The raw GAN uses its fixed final epoch. Supervised checkpoints use the highest validation ROC-AUC, with earlier epochs winning ties.

| Model | Validation ROC-AUC | Selected epoch |
| :--- | ---: | ---: |
{valrows}

### Final Holdout Evaluation (4,000 Images)

The primary test contains 2,000 real and 2,000 Stable Diffusion images. Positive class = AI-generated; the decision threshold was fixed at 0.5 before evaluation.

| Model | ROC-AUC | Macro-F1 | Accuracy |
| :--- | ---: | ---: | ---: |
{testrows}

The selected detector's ROC-AUC has a conditional 95% bootstrap interval of **[{ci[0]:.4f}, {ci[1]:.4f}]**. The paired ROC-AUC difference, GAN-initialized minus scratch, is **{delta['point']:+.4f}**, with interval **[{delta['paired_95_percentile_interval'][0]:+.4f}, {delta['paired_95_percentile_interval'][1]:+.4f}]**. These are 500 class-stratified resamples of the same held-out predictions. They do not measure variability across training seeds, datasets, or generators, and assume the sampled images are sufficiently independent.

### Generator Transfer and Compression Stress Test

| Model | CIFAKE ROC-AUC | JPEG Q50 ROC-AUC | Own GAN ROC-AUC |
| :--- | ---: | ---: | ---: |
{robustness}

JPEG recompression changes the selected detector's accuracy from **{100*best['accuracy']:.2f}% to {100*stress['accuracy']:.2f}%**, and its false-positive rate from **{100*best['false_positive_rate']:.2f}% to {100*stress['false_positive_rate']:.2f}%**. The same real test images are reused across conditions; these are related evaluations, not independent replications.

### Experimental Design

1. **Adversarial pretraining:** A generator maps 64-dimensional Gaussian noise to 32 × 32 RGB images. A convolutional discriminator learns real = 0 and generated = 1. Both train jointly for 10 epochs on the 6,000 real training images only.
2. **Raw discriminator:** Freeze the GAN's final discriminator. No Stable Diffusion images are used to train this model.
3. **Scratch detector:** Train the same discriminator architecture from random initialization on 6,000 real and 6,000 Stable Diffusion training images for six epochs.
4. **GAN-initialized detector:** Initialize that architecture with the GAN discriminator's learned weights, then fine-tune all weights on the same labeled images for the same six supervised epochs.
5. **Evaluation:** Select supervised checkpoints and the reported model using 2,000 validation images before scoring the test set. Also evaluate JPEG quality 50 and 2,000 fresh GAN outputs against the same 2,000 real test images. GAN test images are quantized and JPEG-encoded at quality 95 before evaluation.

All three discriminators contain 167,649 parameters: three stride-2 convolutional layers (32/64/128 channels), leaky ReLU, dropout 0.25, and a dense output. The generator has 305,027 parameters and uses transpose convolutions, batch normalization, ReLU, and tanh. Input pixels are decoded to RGB and scaled to [-1, 1]. Batch size is 128; seed is 42. GAN Adam uses learning rate 0.0002 and beta1 = 0.5; supervised Adam uses 0.001. Gradient global norm is clipped at 1.0.

The two supervised models share architecture, data, and supervised epoch budget. The GAN-initialized model also receives adversarial pretraining, so total compute is not matched. No pretrained external weights or paid model APIs are used. The GAN receives a short fixed training budget; its displayed outputs are highly repetitive checkerboard patterns, not realistic photographs. Its near-perfect own-generator detection is therefore an easy artifact-detection result. GAN losses alone do not establish visual quality, and this unsuccessful run does not establish that GAN pretraining generally cannot help.

---

## Data Summary

* **Source:** [CIFAKE by Jordan J. Bird and Ahmad Lotfi](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images), pinned commit `e112a942abaecd02b6b1f6f646c807d56be8fb62`. Real images originate from CIFAR-10; synthetic images were generated by the dataset authors using Stable Diffusion 1.4.
* **Data Quality:** 18,000 readable RGB images; 24 exact-duplicate candidates skipped; 18,000 unique decoded-pixel hashes. No pixel imputation. Near duplicates and shared content are not ruled out.
* **Partitions:** 12,000 training, 2,000 validation, and 4,000 test images, balanced by real/AI label. Training and validation come only from the source training split; test images come only from its test split.
* **Selection:** Within each original split/class, sort SHA-256 of `42:path` and retain the first images with unique decoded-pixel hashes. The first 6,000 retained training images per class form training and the next 1,000 form validation. Paths and both file/pixel hashes are recorded in `sample_manifest.csv.gz`.
* **Integrity:** All 18,000 decompressed manifest records were checked against the pinned author archive. Original images and the source archive are referenced, not republished as a dataset snapshot.
* **Interpretability:** Error examples show failure cases. They are not causal explanations or evidence of which visual features the model uses; tabular permutation importance and feature correlations are not reported.

### Error Analysis and Limits

At threshold 0.5, the selected detector falsely flags **{best['confusion_matrix'][0][1]} real images** and misses **{best['confusion_matrix'][1][0]} synthetic images**. Every selected-model error is recorded in `error_analysis.csv`; all 42,000 model scores across validation and test conditions are in `predictions.csv`.

The classifier can exploit generator, resizing, or compression artifacts. It has only been tested on a small 32 × 32 benchmark, this one locally trained GAN, and one diffusion-model source. Its scores are not calibrated probabilities of authenticity. High-resolution photographs, screenshots, newer generators, edited images, and different class balances need separate evaluation. This study does not establish a reliable detector for arbitrary online pictures.

### Artifact Verification

Saved classifier checkpoints are checked against all recorded validation and test-condition scores. Regenerating the 2,000 GAN test images produced 1,989 byte-identical images and small pixel differences in 11 images (maximum RGB-channel difference 5/255; mean absolute difference 0.000256 on the 0–255 scale). The original experimental outputs are preserved in `generated_test_images.npz` for exact own-GAN evaluation. This archive contains generated experimental images, not source CIFAKE images. Its complete decompressed array is checked against the original run. All 18,000 compressed source-manifest records are separately checked against the public author archive.

---

## Visualizations

| Dataset examples | GAN training progression |
| :---: | :---: |
| ![Dataset examples](figures/dataset-examples.png) | ![GAN progression](figures/gan-progress.png) |

| Training curves | Primary model comparison |
| :---: | :---: |
| ![Learning curves](figures/learning-curves.png) | ![Model comparison](figures/model-comparison.png) |

| ROC curves | Confusion matrices |
| :---: | :---: |
| ![ROC curves](figures/roc-curve.png) | ![Confusion matrices](figures/confusion-matrix.png) |

| Transfer and compression sensitivity | Selected detector errors |
| :---: | :---: |
| ![Robustness](figures/robustness.png) | ![Error examples](figures/error-examples.png) |

---

## Repository Structure

```text
├── figures/                    # Eight actual diagnostic and comparison plots
├── analysis.ipynb              # Executed analysis of saved predictions
├── prepare_data.py             # Pinned download, selection, and integrity checks
├── train.py                    # GAN, detector training, and fixed evaluations
├── verify.py                   # Full checkpoint and source-manifest verification
├── build_report.py             # Figures, uncertainty analysis, and report
├── predict.py                  # Score an image using a saved detector
├── requirements.txt            # Pinned direct dependencies
├── audit.json                  # Actual training environment, dates, and hashes
├── verification_environment.txt # Environment used to reload and verify models
├── data_sources.json           # Archive identity, counts, and selection policy
├── sample_manifest.csv.gz      # Every selected source path and integrity hash
├── duplicates_removed.json     # Skipped duplicate candidates
├── study_config.json           # Frozen training/evaluation choices
├── training_history.csv        # Actual GAN losses and detector validation scores
├── selection.json              # Validation-only checkpoint/model choices
├── predictions.csv             # All validation and test-condition scores
├── generated_test_images.npz    # Original 2,000 synthetic test outputs
├── error_analysis.csv          # Selected detector's 516 primary-test errors
├── metrics.json                # Full validation and holdout metrics
├── uncertainty.json            # Conditional bootstrap intervals
├── *.weights.npz               # Generator and all three detector checkpoints
├── verification.json           # Full prediction and source-check results
├── DATA_LICENSE.txt            # Dataset authors' license statement
├── SOURCE.md                   # Dataset and research references
├── LEARNING_NOTES.md           # Unfilled human review prompts
├── repository_metadata.json    # Suggested standalone name, About, and topics
├── report.html                 # Report with relative links to figure files
├── project.json                # Completed-study manifest and artifact hashes
└── README.md
```

Training ran on 28 September 2026; documentation and publication followed on 29 September. Automated implementation, execution, and analysis by OpenAI Codex for Saud Alotaibi. Human review is pending; personal learning answers remain unfilled.
'''
    (P/'README.md').write_text(readme)
    save('repository_metadata.json',dict(suggested_repo_name='tensorflow-gan-image-detection',about='Test GAN discriminator transfer to AI-image detection with TensorFlow, CIFAKE, verified checkpoints, compression stress tests, and error analysis.',topics=['tensorflow','gan','dcgan','ai-image-detection','cifake','deep-learning','computer-vision']))
    (P/'SOURCE.md').write_text('''# Sources and attribution

- Bird, J. J. and Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access. https://arxiv.org/abs/2303.14126
- Author dataset repository: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/tree/e112a942abaecd02b6b1f6f646c807d56be8fb62
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- Goodfellow et al. (2014). Generative Adversarial Nets. https://arxiv.org/abs/1406.2661
- Radford, Metz, and Chintala (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. https://arxiv.org/abs/1511.06434
- TensorFlow DCGAN tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

CIFAKE's authors distribute the dataset under an MIT license statement, retained in DATA_LICENSE.txt. The complete source archive was fetched from the pinned author commit; archive SHA-256 and per-image file/pixel hashes are recorded. Only small illustrative figures are republished, not the dataset archive. Stable Diffusion images are supplied by the dataset authors; this study did not run Stable Diffusion.

The GAN and detector code was independently written for this controlled experiment. A GAN discriminator learns to distinguish its training generator from its real distribution; this is not automatically a universal AI-image detector.
''')
    (P/'LEARNING_NOTES.md').write_text('# Human learning review\n\n## What does the discriminator learn?\n\n## Why might performance fail on another generator?\n\n## How did GAN initialization change the result?\n\n## What would I test before using this on real-world photos?\n\n')
    codes=["import json\nfrom pathlib import Path\nimport pandas as pd\np = Path('.')\nm = json.loads((p / 'metrics.json').read_text())\nprint('Selected on validation:', m['selected_model'])", "rows = [{'model': n, **{k: r['test']['CIFAKE holdout'][k] for k in ['roc_auc','macro_f1','accuracy']}} for n,r in m['models'].items()]\nprint(pd.DataFrame(rows).to_string(index=False))", "e = pd.read_csv(p / 'error_analysis.csv')\nprint('Primary-test errors:', len(e))\nprint(e.groupby('label').size().to_string())", "for n,r in m['models'].items():\n    print(n, {condition: round(v['roc_auc'], 4) for condition,v in r['test'].items()})"]
    cells=[dict(cell_type='markdown',metadata={},source=['# GAN transfer and AI-image detection\n','Executed analysis of saved artifacts; training is implemented in train.py.'])]; namespace={}; previous=os.getcwd(); os.chdir(P)
    for i,code in enumerate(codes,1):
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer): exec(compile(code,f'cell-{i}','exec'),namespace)
        cells.append(dict(cell_type='code',metadata={},execution_count=i,source=code.splitlines(keepends=True),outputs=[dict(output_type='stream',name='stdout',text=buffer.getvalue().splitlines(keepends=True))]))
    os.chdir(previous)
    for i,cell in enumerate(cells): cell['id']=f'gan-analysis-{i}'
    save('analysis.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'},'execution_method':'Python in-process cell execution with captured stdout'},nbformat=4,nbformat_minor=5))
    table=pd.DataFrame([dict(model=n,roc_auc=r['test'][conditions[0]]['roc_auc'],macro_f1=r['test'][conditions[0]]['macro_f1'],accuracy=r['test'][conditions[0]]['accuracy']) for n,r in models.items()]).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body='<h1>Can a GAN Discriminator Detect AI-Generated Images?</h1><p>A TensorFlow study of transfer from a small DCGAN to CIFAKE diffusion-image detection.</p>'+table+'<p>GAN pretraining did not improve the primary test result in this one-seed study. Generalization beyond this small benchmark is unproven. See README.md for the full protocol and limitations.</p>'
    for f in sorted(figs.glob('*.png')): body+=f'<figure><img src="figures/{f.name}" alt="{html.escape(f.stem)}"></figure>'
    (P/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>GAN image-detection study</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}td,th{padding:10px;text-align:left}figure{margin:24px 0}</style>'+body+'</html>')
    print(json.dumps({'selected':chosen,'errors':len(mistakes),'uncertainty':uncertainty},indent=2))
if __name__=='__main__': main()
