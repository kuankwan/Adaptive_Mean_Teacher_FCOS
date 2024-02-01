import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy


def visualize_grid_attention_v2(img, save_path, attention_mask,index,level, ratio=1, cmap="jet", save_image=False,
                                save_original_image=True, quality=300):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    img = img.cpu().numpy().transpose(1, 2, 0)
    # 反Normalize操作
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    img_h, img_w,_ = img.shape
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # scale the image
    img_h, img_w = int(img_h * ratio), int(img_w * ratio)
    # img = img.resize((img_h, img_w))
    plt.imshow(img.astype('uint8'), alpha=1)
    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = "original_pic.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        plt.axis('off')
        # plt.savefig(original_image_save_path, quality=quality)
        cv2.imwrite(original_image_save_path,img)
    # plt.imshow(img, alpha=1)
    plt.axis('off')
    # tt_mask = cv2.cvtColor(attention_mask,cv2.COLOR_RGB2BGR)
    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_w, img_h))
    # normed_mask = mask / mask.max()
    normed_mask = (mask * 255).astype('uint8')

    plt.imshow(normed_mask, alpha=0.3, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_name =  "level%d_cls%d_with_attention.jpg"%(level,index)
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path,format='png',pad_inches = 0,transparent=True, dpi=quality)
    plt.close()

def visualize_mask(img, save_path, attention_mask,level, ratio=1, save_image=False,
                                save_original_image=True, quality=300):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    img = img.cpu().numpy().transpose(1, 2, 0)
    # 反Normalize操作
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    img_h, img_w,_ = img.shape
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # scale the image
    img_h, img_w = int(img_h * ratio), int(img_w * ratio)
    # img = img.resize((img_h, img_w))
    plt.imshow(img.astype('uint8'), alpha=1)
    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = "original_pic.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        plt.axis('off')
        # plt.savefig(original_image_save_path, quality=quality)
        cv2.imwrite(original_image_save_path,img)
    # plt.imshow(img, alpha=1)
    plt.axis('off')
    # tt_mask = cv2.cvtColor(attention_mask,cv2.COLOR_RGB2BGR)
    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_w, img_h))
    # normed_mask = mask / mask.max()
    # normed_mask = (mask * 255).astype('uint8')
    # img = cv2.add(img,np.zeros(np.shape(img),dtype=np.uint8),mask=mask)
    plt.imshow(mask.astype('uint8'), alpha=0.3)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_name =  "level%d_cls_with_mask.jpg"%(level)
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path,format='png',pad_inches = 0,transparent=True, dpi=quality)
    plt.close()




