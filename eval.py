import sys
sys.path.append("..") # Adds higher directory to python modules path.

import torch
from torch.utils.data import DataLoader
from utils.utils import *
import argparse
import json
from utils.datasets import ScutHeadDataset, VOCDetection
from models import Darknet
from tqdm import tqdm
from utils.plot import draw_prediction


def _evaluate_on_single_dataset(model, val_loader, val_config):
    model.eval()

    conf_thres = val_config["conf_threshold"]
    nms_thres = val_config["nms_threshold"]
    iou_thres = val_config["iou_threshold"]

    # FIXME: move somewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger.info(f"Epoch {epoch} started!")
    sample_metrics = []
    labels = []

    bar = tqdm(val_loader, desc="Evaluating ")
    for i, (image_batch, bboxes) in enumerate(bar):
        image_batch = image_batch.to(device)
        # bboxes = bboxes.to(device)

        img_size = image_batch.shape[2]

        labels += bboxes[:, 1].tolist()

        # Rescale target
        bboxes[:, 2:] = xywh2xyxy(bboxes[:, 2:])
        bboxes[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(image_batch)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # draw_prediction(
        #     image_batch[0].cpu().int().permute(1,2,0),
        #     outputs[0][:, :4],
        #     bboxes[bboxes[:, 0] == 0][:, -4:])

        sample_metrics += get_batch_statistics(outputs, bboxes, iou_threshold=iou_thres)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def evaluate(model, val_loader_dict, val_config):
    """

    :param model:
    :param val_loader_dict: dict of {dataset name: dataset DataLoader}
    :param val_config: 'val' part of JSON-config
    :return:
    """
    result_dict = dict()
    for name, dataloader in val_loader_dict.items():
        precision, recall, AP, f1, ap_class = _evaluate_on_single_dataset(model, dataloader, val_config)

        res = dict()
        res['precision'], res['recall'], res['AP'], res['F1'], res['AP_class'] = precision, recall, AP, f1, ap_class
        result_dict[name] = res

    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="runs/config.json")
    parser.add_argument("--output_dir", default='output')
    parser.add_argument("--model_checkpoint")
    args = parser.parse_args()

    with open(args.config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    val_loader_dict = dict()
    for i, dataset_config in enumerate(config['val']["datasets"]):
        val_dataset = VOCDetection(img_dir=dataset_config["image_folder"],
                                   annotation_dir=dataset_config["annot_folder"],
                                   cache_dir=dataset_config["cache_dir"],
                                   split_file=dataset_config['split_file'],
                                   img_size=config['model']['input_size'],
                                   filter_labels=config['model']['labels'],
                                   multiscale=False,
                                   augment=False)
        val_dataset.name = dataset_config.get('name')

        val_loader = DataLoader(val_dataset,
                                batch_size=config["val"]["batch_size"],
                                collate_fn=val_dataset.collate_fn,
                                shuffle=True)
        dataset_name = val_dataset.name if val_dataset.name else f"Dataset #{i}"
        val_loader_dict[dataset_name] = val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(config["model"]["config"]).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.eval()

    result_dict = evaluate(model, val_loader_dict, config["val"])
    for name, results in result_dict.items():
        output_str = f"{name} evaluation results:\n" \
            f"precision-{results['precision']},\n" \
            f"recall-{results['recall']},\n" \
            f"AP-{results['AP']},\n" \
            f"F1-{results['F1']},\n" \
            f"ap_class-{results['AP_class']}"
        print(output_str)

if __name__ == '__main__':
    main()

