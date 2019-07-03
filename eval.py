import sys
sys.path.append("..") # Adds higher directory to python modules path.

import torch
from torch.utils.data import DataLoader
from utils.utils import *
import argparse
import json
from utils.datasets import ScutHeadDataset
from models import Darknet
from tqdm import tqdm
from utils.plot import draw_prediction


def evaluate(model, val_dataloader, val_config):
    model.eval()

    conf_thres = val_config["conf_threshold"]
    nms_thres = val_config["nms_threshold"]
    iou_thres = val_config["iou_threshold"]

    # FIXME: move somewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger.info(f"Epoch {epoch} started!")
    sample_metrics = []
    labels = []

    bar = tqdm(val_dataloader, desc="Evaluating ")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="runs/config.json")
    parser.add_argument("--output_dir", default='output')
    parser.add_argument("--model_checkpoint")
    args = parser.parse_args()

    with open(args.config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    val_dataset = ScutHeadDataset(img_dir=config['val']["image_folder"],
                                  annotation_dir=config['val']["annot_folder"],
                                  cache_dir=config['val']["cache_dir"],
                                  split_file=config['val']['split_file'],
                                  img_size=config['model']['input_size'],
                                  filter_labels=config['model']['labels'],
                                  multiscale=False,  # TODO: not sure if it should be False
                                  augment=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config["val"]["batch_size"],
                            collate_fn=val_dataset.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(config["model"]["config"]).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.eval()

    precision, recall, AP, f1, ap_class = evaluate(model, val_loader, config["val"])
    output_str = f"Evaluating results: precision-{precision}, recall-{recall}, AP-{AP}, F1-{f1}, ap_class-{ap_class}"
    print(output_str)


if __name__ == '__main__':
    main()

