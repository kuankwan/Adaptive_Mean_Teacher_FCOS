# Cross-domain object detection
This project is my library of **One-stage Object Detection**.
My code is clean and concise, without too deep encapsulation, 
so it is easy to understand the function of each module.



# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n detection python=3.7
```

- Then, activate the environment:
```Shell
conda activate detection
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

We suggest that PyTorch should be higher than 1.7.1. 
At least, please make sure your torch is version 1.x.

In my FCOS:
- For regression head, `GIoU loss` is deployed rather than `IoU loss`
- For real-time FCOS, the `PaFPN` is deployed for fpn

# Train
## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

## Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolof50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show



