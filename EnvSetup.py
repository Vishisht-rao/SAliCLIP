import os


if not os.path.isdir('taming-transformers'):
    os.system("git clone https://github.com/CompVis/taming-transformers")
 
if not os.path.isfile("coco.yaml"):
    os.system("curl -L -o coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml'")   
    
if not os.path.isfile("coco.ckpt"):
    os.system("curl -L -o coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'")
