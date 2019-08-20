import numpy as np
import copy


def bbox_iou_numpy(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
        inter_rect_y2 - inter_rect_y1 + 1, 0, None
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class Tracklet:
    """
    
    bbox - numpy array with (7, ) shape. x1, y1, x2, y2, obj_conf, class_conf, class
    """

    def __init__(self, init_bbox):
        self.bboxes = [init_bbox]
        self.max_score = init_bbox[4]
        self.color = np.random.rand(3,) * 255
        self.death_count = 0

    def new_bbox(self, bbox, drop_death_count=True):
        if bbox[4] > self.max_score:
            self.max_score = bbox[4]

        self.bboxes.append(bbox)

        if drop_death_count:
            self.death_count = 0

    @property
    def last_bbox(self):
        return self.bboxes[-1]
    
    @property
    def length(self):
        return len(self.bboxes)
    
    def update(self, boxes, sigma_iou):
        
        was_updated, best_match_ind = False, -1
        ious = bbox_iou_numpy(
            np.repeat(self.last_bbox[None, :], boxes.shape[0], axis=0),
            boxes
        )
        best_match_ind = ious.argmax()
        best_match, best_match_iou = boxes[best_match_ind], ious[best_match_ind]
        if best_match_iou >= sigma_iou:
            self.new_bbox(best_match)
            was_updated = True
            
        return was_updated, best_match_ind
    
class TrackingState:
    """
    Class that contains list of current active tracks
    """
    
    def __init__(self, sigma_h=0.5, sigma_iou=0.25, min_length=3, max_lost_time=5):
        
        self.tracks = []
        self.finished_tracks = []
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.min_length = min_length
        self.max_lost_time = max_lost_time
        
    @property
    def active_tracks(self):
        return [track for track in self.tracks if track.length >= self.min_length]
        
    def update_tracks(self, boxes):
        """
        
        boxes - numpy array with (N, 7) shape
        """

        boxes_temp = boxes.copy() # we need to remove bboxes from list later
        updated_tracks = []
        lost_tracks = []
        for track in self.tracks:

            was_updated = False
            if len(boxes_temp) > 0:
                was_updated, best_match_ind = track.update(boxes_temp, self.sigma_iou)

                if was_updated:
                    updated_tracks.append(track)
                    # remove from best matching detection from detections
                    boxes_temp = np.delete(boxes_temp, best_match_ind, axis=0)
                    # del boxes_temp[best_match_ind]

            # if track was not updated
            # if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
            if not was_updated:
                # finish track when the conditions are met
                if track.max_score >= self.sigma_h and len(track.bboxes) >= self.min_length:
                    lost_tracks.append(track)
                    track.death_count += 1

        # create new tracks from detection that left
        new_tracks = [Tracklet(det)
                      for det in boxes_temp if det[4] >= self.sigma_h]

        # remove long lost tracks
        finished_tracks = [track for track in lost_tracks if track.death_count > self.max_lost_time]
        lost_tracks = [track for track in lost_tracks if track not in finished_tracks]
        self.finished_tracks += finished_tracks
        # extend lost tracks by same last bbox
        [track.new_bbox(track.last_bbox, drop_death_count=False) for track in lost_tracks]

        active_boxes = [track.last_bbox for track in self.tracks]
        active_tracks = updated_tracks + new_tracks + lost_tracks
        
#         print(f'updated: {len(updated_tracks)}')
#         print(f'new    : {len(new_tracks)}')
#         print(f'lost   : {len(lost_tracks)}')

        self.tracks = active_tracks
        # return active_tracks
