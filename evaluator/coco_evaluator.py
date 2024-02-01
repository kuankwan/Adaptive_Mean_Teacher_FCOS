import json
import tempfile
import torch
from dataset.coco import *
import time
from utils.visual_attention import visualize_grid_attention_v2
try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
        if self.testset:
            image_set = 'test2017'
        else:
            image_set = 'val'

        self.dataset = COCODataset(
                            data_dir=data_dir,
                            image_set=image_set,
                            transform=None)
        self.transform = transform
        self.device = device

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.

    def evaluate(self, model,logger=None):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        voc_map_info_list = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        num_warmup = 5
        pure_inf_time = 0
        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            # load an image
            img, id_ = self.dataset.pull_image(index)
            h, w, _ = img.shape
            orig_size = np.array([[w, h, w, h]])
            img_path = self.dataset.get_image_path(index)
            # preprocess
            x = self.transform(img)[0]
            x = x.unsqueeze(0).to(self.device)
            
            id_ = int(id_)
            ids.append(id_)
            # inference
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                outputs = model(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                if index >= num_warmup:
                    pure_inf_time += elapsed
                if (index + 1) == num_images:
                    fps = (index + 1 - num_warmup) / pure_inf_time
                    print(
                        f'Overall fps: {fps:.1f} img / s, '
                        f'times per image: {1000 / fps:.1f} ms / img',
                        flush=True)
                    if logger is not None:
                        logger.info('[Eval: %d / %d]'%(index, num_images))
                        logger.info(
                            f'Overall fps: {fps:.1f} img / s, '
                            f'times per image: {1000 / fps:.1f} ms / img',
                        )
                bboxes, scores, cls_inds = outputs
                # visualize_grid_attention_v2(img_path,'./attention',cls_mask.sigmoid(),save_image=True)
                # rescale
                if self.transform.padding:
                    # The input image is padded with 0 on the short side, aligning with the long side.
                    bboxes *= max(h, w)
                else:
                    # the input image is not padded.
                    bboxes *= orig_size

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('coco_test-dev.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_test-dev.json')
                return -1, -1
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
                cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                cocoEval.params.imgIds = ids
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                loginfo = {
                    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': cocoEval.stats[0],
                    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]': cocoEval.stats[1],
                    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]': cocoEval.stats[2],
                    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': cocoEval.stats[3],
                    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': cocoEval.stats[4],
                    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': cocoEval.stats[5],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]': cocoEval.stats[6],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]': cocoEval.stats[7],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': cocoEval.stats[8],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': cocoEval.stats[9],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': cocoEval.stats[10],
                    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': cocoEval.stats[11],
                }
                if logger is not None:
                    for key, value in loginfo.items():
                        log_info = f'{key}: {value}'
                        logger.info(log_info)
                ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                print('ap50_95 : ', ap50_95)
                print('ap50 : ', ap50)
                for i in range(len(self.dataset.coco.cats)):
                    stats, _ = summarize(cocoEval, catId=i)
                    voc_map_info_list.append(" {:15}: {}".format(self.dataset.coco.cats[i+1]["name"], stats[1]))
                print_coco = "\n".join(voc_map_info_list)
                if logger is not None:
                    logger.info(print_coco)
                print(print_coco)
                self.map = ap50_95
                self.ap50_95 = ap50_95
                self.ap50 = ap50

                return ap50, ap50_95
        else:
            return 0, 0

