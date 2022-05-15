# (CVPR 2021) Human De-occlusion: Invisible Perception and Recovery for Humans

## Abstract

In this paper, we tackle the problem of human de- occlusion which reasons about occluded segmentation masks and invisible appearance content of humans. In particular, a two-stage framework is proposed to estimate the invisible portions and recover the content inside. For the stage of mask completion, a stacked network structure is devised to refine inaccurate masks from a general instance segmentation model and predict integrated masks simultaneously. Additionally, the guidance from human parsing and typical pose masks are leveraged to bring prior information. For the stage of content recovery, a novel parsing guided attention module is applied to isolate body parts and capture context information across multiple scales. Besides, an Amodal Human Perception dataset (AHP) is collected to settle the task of human de-occlusion. AHP has advantages of providing annotations from real-world scenes and the number of humans is comparatively larger than other amodal perception datasets. Based on this dataset, experiments demonstrate that our method performs over the state-of-the-art techniques in both tasks of mask completion and content recovery. Our AHP dataset is available at https://sydney0zq.github.io/ahp/.

Note: This code only contains the core implementation of the model definition. And it is just for reference, not guarantee to be fully correct. And I will update the code to make it runnable if necessary.


## File Structure

```
src
└── models
    ├── c
    │   ├── backbone
    │   │   ├── discriminator.py
    │   │   ├── others.py
    │   │   ├── pconv_unet.py
    │   │   ├── resnet.py
    │   │   ├── resnet_cls.py
    │   │   ├── unet
    │   │   │   ├── __init__.py
    │   │   │   ├── unet_model.py
    │   │   │   ├── unet_parts.py
    │   │   │   └── unet_resnet_model.py
    │   │   └── vae.py
    │   ├── gan_model.py
    │   ├── losses.py
    │   ├── partial_completion_content_cgan.py
    │   └── single_stage_model.py
    └── m
        ├── backbone
        │   ├── __init__.py
        │   ├── centers
        │   │   └── centers_64.pth
        │   ├── discriminator.py
        │   ├── inception.py
        │   ├── pconv_unet.py
        │   ├── resnet.py
        │   └── unet.py
        ├── dice_loss.py
        ├── dis.py
        ├── losses.py
        ├── lovasz_losses.py
        ├── metrics.py
        ├── others.py
        ├── partial_completion_mask.py
        └── single_stage_model.py
```


## Citation

```
@inproceedings{zhou2021human,
  title={Human De-occlusion: Invisible Perception and Recovery for Humans},
  author={Zhou, Qiang and Wang, Shiyin and Wang, Yitong and Huang, Zilong and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3691--3701},
  year={2021}
}
```

## Acknowledgement

The code is heavily borrow from https://github.com/XiaohangZhan/deocclusion. Thanks for their great works!