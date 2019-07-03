import os
import numpy as np
from utils.datasets import ScutHeadDataset
from eval import evaluate
from models import Darknet
from utils.utils import *
from utils.plot import draw_image_batch_with_targets
from tensorboardX import SummaryWriter
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import logging
import argparse
import shutil
import sys


def get_experiment_name(config):
    batch_size = config["train"]["batch_size"]
    grad_accum = config["train"]["gradient_accumulations"]
    learning_rate = config["train"]["learning_rate"]

    augment = "aug__" if config["train"]["augment"] else ""
    grad_clipping = "grad_clip__" if config["train"]["gradient_clipping"] else ""
    frozen_extractor = "frozen__" if config["train"]["freeze_feature_extractor"] else ""

    return f"exp__b_{batch_size}_grad_{grad_accum}" \
        f"__lr_{learning_rate:.1e}__{augment}{frozen_extractor}{grad_clipping}"


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm


def prepare_dataloaders(config):
    train_dataset = ScutHeadDataset(img_dir=config['train']["image_folder"],
                                    annotation_dir=config['train']["annot_folder"],
                                    cache_dir=config['train']["cache_dir"],
                                    split_file=config['train']['split_file'],
                                    img_size=config['model']['input_size'],
                                    filter_labels=config['model']['labels'],
                                    multiscale=True,
                                    augment=config['train']['augment'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config["train"]["batch_size"],
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True)

    if not config['val']['validate']:
        return train_loader

    else:
        val_dataset = ScutHeadDataset(img_dir=config['val']["image_folder"],
                                      annotation_dir=config['val']["annot_folder"],
                                      cache_dir=config['val']["cache_dir"],
                                      split_file=config['val']['split_file'],
                                      img_size=config['model']['input_size'],
                                      filter_labels=config['model']['labels'],
                                      multiscale=False,
                                      augment=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=config["val"]["batch_size"],
                                collate_fn=val_dataset.collate_fn,
                                shuffle=True)

        return train_loader, val_loader


metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/runs/config.json")
    parser.add_argument("--output_dir" , default='output')
    args = parser.parse_args()

    with open(args.config_file) as config_buffer:
        config = json.loads(config_buffer.read())

    exp_name = get_experiment_name(config)
    print(f"Experiment name: {exp_name}")
    out_dir = os.path.join(args.output_dir, exp_name)
    if os.path.exists(out_dir):
        print("experiment dir already exists! Removing...")
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    log_dir = f"{out_dir}/logs"
    checkpoint_dir = f"{out_dir}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tb_logger = SummaryWriter(log_dir)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[
                            logging.FileHandler(f"{out_dir}/log.log"),
                            logging.StreamHandler(sys.stdout)
                        ],
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    logging.info("New session")

    seed = config["train"]["seed"]
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)

    ###############################
    #   Prepare data loaders
    ###############################
    print("Loading datasets...")
    if config['val']['validate']:
        train_loader, val_loader = prepare_dataloaders(config)
    else:
        train_loader = prepare_dataloaders(config)
    print("Loaded!")
    if config["train"]["debug"]:
        image_batch, target = next(iter(train_loader))
        draw_image_batch_with_targets(image_batch[:4], target, cols=2)

        if config['val']['validate']:
            val_image_batch, val_target = next(iter(val_loader))
            draw_image_batch_with_targets(val_image_batch[:4], val_target, cols=2)

    ###############################
    #   Construct the model
    ###############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(config["model"]["config"]).to(device)
    model.apply(weights_init_normal)
    print("Model initialized!")

    if config["train"]["freeze_feature_extractor"]:
        model.freeze_feature_extractor()

    print(f"Trainable params: {get_trainable_params_num(model):,}")

    # If specified we start from checkpoint
    if config["model"]["pretrained_weights"]:
        if config["model"]["pretrained_weights"].endswith(".pth"):
            model.load_state_dict(torch.load(config["model"]["pretrained_weights"]))
        else:
            model.load_darknet_weights(config["model"]["pretrained_weights"])
            print("Pretrained weights loaded!")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["train"]["learning_rate"])

    ###############################
    #   Training
    ###############################
    batches_done = 0
    grad_accumulations = config["train"]["gradient_accumulations"]
    save_every = config["train"]["save_every"]

    if config["val"]["validate"]:
        val_iterator = iter(val_loader)

    for epoch in range(config["train"]["nb_epochs"]):

        effective_loss = 0
        loss_history = torch.zeros(len(train_loader))
        logger.info(f"Epoch {epoch} started!")
        bar = tqdm(train_loader)
        for i, (image_batch, bboxes) in enumerate(bar):
            model.train()
            image_batch = image_batch.to(device)
            bboxes = bboxes.to(device)

            loss, outputs = model(image_batch, bboxes)
            effective_loss += loss.item()
            loss_history[i] = loss.item()

            loss.backward()

            if i % grad_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()

                if config["train"]["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                grad_norm = get_grad_norm(model)

                optimizer.zero_grad()
                if config["val"]["validate"]:
                    model.eval()

                    try:
                        val_image_batch, val_bboxes = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        val_image_batch, val_bboxes = next(val_iterator)
                    val_image_batch = val_image_batch.to(device)
                    val_bboxes = val_bboxes.to(device)
                    with torch.no_grad():
                        val_loss, val_outputs = model(val_image_batch, val_bboxes)

                    tb_logger.add_scalar("loss/validation", val_loss, batches_done)

                bar.set_description(f"Loss: {effective_loss / grad_accumulations:.6f}")

                batches_done += 1

                # Tensorboard logging
                for metric_name in metrics:
                    metric_dict = {}
                    for j, yolo_layer in enumerate(model.yolo_layers):
                        metric_dict[f"yolo_{j}"] = yolo_layer.metrics[metric_name]

                    if metric_name == 'loss':
                        metric_dict["overall"] = loss.item()

                    tb_logger.add_scalars(metric_name, metric_dict, batches_done)
                tb_logger.add_scalar("grad_norm", grad_norm, batches_done)
                tb_logger.add_scalar("loss/effective_loss", effective_loss, batches_done)

                effective_loss = 0

                # save model
                if save_every > 0 and batches_done % save_every == 0:
                    torch.save(model.state_dict(), f"{checkpoint_dir}/yolov3_{batches_done}.pth")

        epoch_loss = loss_history.mean()
        print(f"Epoch loss: {epoch_loss}")
        tb_logger.add_scalar("epoch_loss", epoch_loss, epoch)


        if config["val"]["validate"]:
            precision, recall, AP, f1, ap_class = evaluate(model, val_loader, config["val"])
            output_str = f"Evaluating results: precision-{precision}, recall-{recall}, AP-{AP}, F1-{f1}, ap_class-{ap_class}"
            logging.info(output_str)
            print(output_str)

            tb_logger.add_scalar("val_precision", precision, epoch)
            tb_logger.add_scalar("val_recall", recall, epoch)
            tb_logger.add_scalar("val_F1", f1, epoch)
            tb_logger.add_scalar("val_AP", AP, epoch)

        # save model
        torch.save(model.state_dict(), f"{checkpoint_dir}/yolov3_epoch_{epoch}.pth")


def log_model_validation(model, val_loader, val_config, tb_logger, epoch_num):
    precision, recall, AP, f1, ap_class = evaluate(model, val_loader, val_config)
    output_str = f"Evaluating results: precision-{precision}, recall-{recall}, AP-{AP}, F1-{f1}, ap_class-{ap_class}"
    logging.info(output_str)
    print(output_str)

    tb_logger.add_scalar("val_precision", precision, epoch)
    tb_logger.add_scalar("val_recall", recall, epoch)
    tb_logger.add_scalar("val_F1", f1, epoch)
    tb_logger.add_scalar("val_AP", AP, epoch)


if __name__ == '__main__':
    main()
