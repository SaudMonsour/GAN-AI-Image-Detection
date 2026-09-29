"""Score an image with a saved detector. This benchmark score is not proof of authenticity."""
import argparse,json
from PIL import Image,ImageOps
import numpy as np
from train import P,discriminator,load,norm,scores
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('image'); parser.add_argument('--model',choices=['selected','scratch','gan-initialized','raw-gan'],default='selected'); args=parser.parse_args()
    choice=args.model
    if choice=='selected':
        name=json.loads((P/'selection.json').read_text())['selected_model']
        choice={'CNN from scratch':'scratch','GAN-initialized detector':'gan-initialized','Raw GAN discriminator':'raw-gan'}[name]
    checkpoint={'scratch':'scratch_detector.weights.npz','gan-initialized':'gan_initialized_detector.weights.npz','raw-gan':'gan_discriminator.weights.npz'}[choice]
    with Image.open(args.image) as image:
        image=ImageOps.exif_transpose(image).convert('RGB'); original_size=image.size
        if image.size!=(32,32): image=image.resize((32,32),Image.Resampling.BILINEAR)
        x=norm(np.asarray(image,dtype=np.uint8)[None])
    model=discriminator(); load(model,checkpoint); score=float(scores(model,x)[0])
    print(json.dumps({'model':choice,'fake_score':score,'threshold':.5,'benchmark_prediction':'AI-generated' if score>=.5 else 'real','original_size':original_size,'limitation':'Uncalibrated 32x32 CIFAKE benchmark score; not reliable proof of authenticity for arbitrary photos.'},indent=2))
if __name__=='__main__': main()
