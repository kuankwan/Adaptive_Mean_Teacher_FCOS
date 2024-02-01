import cv2,os
import json
#
# x = [479, 378, 195, 124]
# img_path = './data/sim10k/images/train/3384645.jpg'
# img = cv.imread(img_path)
# cv.rectangle(img,(int(x[0]),int(x[1])),(int(x[0]+x[2]),int(x[1]+x[3])),(255,0,0),3)
# cv.imshow('img',img)
# cv.waitKey(0)
#
# json_file = 'data/kitti/annotations/train_kitti_caronly.json'
# json_dict = {}
# with open(json_file,'r',encoding='utf8') as fp:
#     json_data = json.load(fp)
#     for ann in json_data['annotations']:
#         b = []
#         for i in ann['bbox']:
#             i = i + 0.0
#             b.append(i)
#         ann['bbox'] = b
#     json_dict = json_data
# json_fp = open(json_file, "w")
# json_str = json.dumps(json_dict)
# json_fp.write(json_str)
# json_fp.close()
import numpy as np
from matplotlib import pyplot as plt
# from pycocotools.coco import COCO

train_json = 'D:\\datasets\\ElevatorPerson\\annotations\\instance_ele_hard_val.json'
train_path = 'D:\\datasets\\ElevatorPerson\\images\\val'
coco_class_labels = ('person',)
# coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8]
coco_class_index = [1,]
# coco_class_color = [(99,178,238),(118,218,145),(248,203,127),(248,149,136),(124,214,207),(145,146,171),(148,60,57),(98,76,124)]
coco_class_color = [(99,178,238)]

def visualization_bbox1(num_image, json_path, img_path):  # 需要画的第num副图片， 对应的json路径和图片路径
    with open(json_path) as annos:
        annotation_json = json.load(annos)

    print('the annotation_json num_key is:', len(annotation_json))  # 统计json文件的关键字长度
    print('the annotation_json key is:', annotation_json.keys())  # 读出json文件的关键字
    print('the annotation_json num_images is:', len(annotation_json['images']))  # json文件中包含的图片数量
    # class_colors = [(np.random.randint(255),
    #                  np.random.randint(255),
    #                  np.random.randint(255)) for _ in range(8)]

    image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
    id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

    image_path = os.path.join(img_path, str(image_name).zfill(5))  # 拼接图像路径
    image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
    num_bbox = 0  # 统计一幅图片中bbox的数量

    for i in range(len(annotation_json['annotations'][::])):
        if annotation_json['annotations'][i - 1]['image_id'] == id:
            num_bbox = num_bbox + 1
            x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
            cls_id = annotation_json['annotations'][i - 1]['category_id']
            color = coco_class_color[cls_id-1]
            mess = '%s' % (coco_class_labels[cls_id-1])
            t_size = cv2.getTextSize(mess, 0, fontScale=1, thickness=1)[0]
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            image = cv2.rectangle(image, (int(x), int(y - t_size[1])), (int(x + t_size[0] * 0.4), int(y)), color, -1)
            image = cv2.putText(image, mess, (int(x), int(y - 5)), 0, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    print('The unm_bbox of the display image is:', num_bbox)

    # 显示方式1：用plt.imshow()显示
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #绘制图像，将CV的BGR换成RGB
    save_pth = './det_results/coco/ele_hard/'
    save_pth = os.path.join(save_pth, image_name)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_pth,dpi=300)
    # plt.show() #显示图像


    # 显示方式2：用cv2.imshow()显示
    # cv2.namedWindow(image_name, 0)  # 创建窗口
    # cv2.resizeWindow(image_name, 1000, 1000)  # 创建500*500的窗口
    # cv2.imshow(image_name, image)
    # cv2.imwrite(save_pth, image)
    # cv2.waitKey(0)


if __name__ == "__main__":
    for i in range(2000):
        visualization_bbox1(i, train_json, train_path)
