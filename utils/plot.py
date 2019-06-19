import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def draw_tracking_state(image, tracking_state, trail_length=40, trail_size=None, print_text=True):
    # TODO: improve performance
    
    image_h, image_w, _ = image.shape
    
    for track in tracking_state.active_tracks:
        
        bbox = track.last_bbox
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        
        if trail_size is None:
            trail_size = (ymax - ymin) // 10

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), track.color, 3) # FIXME: track.color
        if print_text:
            cv2.putText(image, 
                        'best score: {0:.4f}'.format(track.max_score),
                        (int(xmin), int(ymin) - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        int(1e-3 * image_h), 
                        track.color, 2)

        for index, bbox in enumerate(track.bboxes[-trail_length:]):

            r = int((index / trail_length) * trail_size)
            
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            xcenter = int( xmin + (xmax - xmin) //2 )
            ycenter = int( ymin + (ymax - ymin) //2 )

            cv2.circle(image, (xcenter, ycenter), int(r), track.color, 5)
        
    return image


def draw_image_batch_with_targets(images, bboxes, cols=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: tensor (B, C, W, H)
    bboxes: tensor (N, 6)

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.

    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """

    n_images = len(images)
    rows = np.ceil(n_images / float(cols))

    _, channels, img_w, img_h = images.shape
    images = images.int().numpy().transpose(0,2,3,1)

    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(rows, cols, n + 1)
        if channels == 1:
            plt.gray()

        image_bboxes = bboxes[bboxes[:, 0] == n]
        for _, _, xc, yc, w, h in image_bboxes:

            x1 = (xc - w * 0.5) * img_w
            y1 = (yc - h * 0.5) * img_h
            w_box = w * img_w
            h_box = h * img_h

            color = (1, 0, 1)
            # Create a Rectangle patch
            patch = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            a.add_patch(patch)

        plt.imshow(image)

    fig.set_size_inches(cols * 4, rows * 4)
    plt.show()


def add_xywh_bboxes(ax, bboxes, img_w, img_h, color=(1, 0, 1)):

    for xc, yc, w, h in bboxes:

        x1 = (xc - w * 0.5) * img_w
        y1 = (yc - h * 0.5) * img_h
        w_box = w * img_w
        h_box = h * img_h

        # Create a Rectangle patch
        patch = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(patch)

    return ax


def add_xyxy_bboxes(ax, bboxes, color=(1, 0, 1)):

    for x1, y1, x2, y2 in bboxes:

        w_box = x2 - x1
        h_box = y2 - y1

        # Create a Rectangle patch
        patch = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(patch)

    return ax


def draw_prediction(img, pred_bboxes, true_bboxes):
    """

    :param img: (W,H,C)
    :param pred_bboxes: (N, 4) x1, y1, x2, y2 ints
    :param true_bboxes: (N, 4) xc, yc, w, h normialized
    :return:
    """

    plt.rcParams['figure.figsize'] = [8, 8]

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    img_w, img_h, _ = img.shape

    add_xywh_bboxes(ax, true_bboxes, img_w, img_h, color=(1,1,0))
    add_xyxy_bboxes(ax, pred_bboxes, color=(1,0,1))

    plt.show()

def draw_image_with_bboxes(img, bboxes):
    """

    :param img: (W,H,C)
    :param bboxes: (N, 4) xc, yc, w, h normialized
    :return:
    """

    plt.rcParams['figure.figsize'] = [8, 8]

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    img_w, img_h, _ = img.shape
    print(img_w, img_h)

    for xc, yc, w, h in bboxes:

        x1 = (xc - w * 0.5) * img_w
        y1 = (yc - h * 0.5) * img_h
        w_box = w * img_w
        h_box = h * img_h

        color = (1, 0, 1)
        # Create a Rectangle patch
        patch = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(patch)

    plt.show()
