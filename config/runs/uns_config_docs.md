# Example with commentaries

```json
{
    "model" : {
        "config": "darknet model config file",
        "pretrained_weights": "path to pre-trained weights file. It can be darknet weights (.weights) or pytorch weights (.pth)",

        "input_size":           "initial input image size",
        "labels":               ["list of tokens. Deprecated"]
    },

    "train": {
        "seed": -1,
        "_seed_comment" : "negative value disables determined seed",

        "batch_size":           16,
        "gradient_accumulations": 4,
        "_effective_batch_size_comment": "effective batch size: batch_size * gradient_accumulations",
        "learning_rate":        1e-4,
        "nb_epochs":            "number of training epochs",

        "augment":              "true|false - apply image augmentation",
        "gradient_clipping": "true|false - clip gradient norm if it is greater then a certain magic number",
        "freeze_feature_extractor": "true|false - freeze backbone layers to speed up training",

        "save_every"        :   "number of training iterations to save model checkpoint. Negative number disables in-epoch checkpoints",

        "debug":                false,

        "datasets": [{
                "name"              :  "SCUT-HEAD",
                "image_folder"      :  "/content/drive/My Drive/Datasets/head-detection/JPEGImages/",
                "annot_folder"      :  "/content/drive/My Drive/Datasets/head-detection/Annotations/",
                "cache_dir"         :  "/content/drive/My Drive/Datasets/head-detection/",
                "split_file"        :  "/content/drive/My Drive/Datasets/head-detection/ImageSets/MainCombined/trainval.txt"
            }]
    },

    "val": {
         "validate"        :  true,

        "nms_threshold"  : "float - non-maximum-suppresion threshold",
        "conf_threshold" : "float - confidience score threshold",
        "iou_threshold"  : "float - IoU threshold",

        "shuffle_dataset": true,

        "batch_size": 16,

        "datasets": [{
            "name"              :  "SCUT-HEAD",
            "image_folder"      :  "/content/drive/My Drive/Datasets/head-detection/JPEGImages/",
            "annot_folder"      :  "/content/drive/My Drive/Datasets/head-detection/Annotations/",
            "cache_dir"         :  "/content/drive/My Drive/Datasets/head-detection/",
            "split_file"        :  "/content/drive/My Drive/Datasets/head-detection/ImageSets/MainCombined/test.txt"
        }, {
            "name"              :  "Hollywood Heads",
            "image_folder"      :  "/content/drive/My Drive/Datasets/HollywoodHeadsTest/JPEGImages/",
            "annot_folder"      :  "/content/drive/My Drive/Datasets/HollywoodHeadsTest/Annotations/",
            "cache_dir"         :  "/content/drive/My Drive/Datasets/HollywoodHeadsTest/",
            "split_file"        :  ""
        }]
    }
}
```

# Datasets list
"datasets" list consists of dataset objects. Each dataset objects must have these fields:

* name - may be some arbitrary string. It used only for representation in logs
* image_folder - path to folder with images
* annot_folder - path to annotations
* cache_dir - path where the annotations cache will be stored
* split_file - path to file consisting of filenames to use for train/validation
