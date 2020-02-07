import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNN

from models import pose_resnet
import argparse
from visualize_keypoints import visualization_keypoints, draw_skeleton
from compute_3d_pose import compute_3d_pose
from voxel_keypoints import voxel_keypoints3
from voxel_keypoints import voxel_keypoints

def heatmaps_to_locs(heatmaps):
    num_images = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    keypoint_locs = np.zeros((num_images, num_keypoints,3))
    for i in range(num_images):
        for j in range(num_keypoints):
            ind = heatmaps[i,j,:,:].argmax().item()
            max_val = heatmaps[i,j,:,:].max().item()
            row, col = np.unravel_index(ind, heatmaps[i,j,:,:].shape)
            val = np.array([col, row, max_val])
            keypoint_locs[i,j,:] = val
    return keypoint_locs

class CropAndPad:

    def __init__(self, out_size=(256,256)):
        self.out_size = out_size[::-1]

    def __call__(self, x):
        image, bb = x
        img_size = image.size
        min_x, min_y = np.round(np.maximum(bb.min(axis=0), np.array([0,0]))).astype(int)
        max_x, max_y = np.round(np.minimum(bb.max(axis=0), np.array(img_size))).astype(int)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max([max_x-min_x, max_y-min_y, self.out_size[0]])
        min_x = center_x - width/2
        min_y = center_y - width/2
        max_x = center_x + width/2
        max_y = center_y + width/2
        image = image.crop(box=(min_x,min_y,max_x,max_y))
        if width != self.out_size[0]:
            image = image.resize(self.out_size)
        return image

class Normalize:

    def __call__(self, image):
        return 2*(image-0.5)


def build_transform():
    '''
    1. transform PIL.Image.RGB to bgr255 as model was finetuned with Caffe2 weights
    2. normailze with coco mean/std
    '''
    to_bgr = T.Lambda(lambda x: x[[2, 1, 0]])
    to_255 = T.Lambda(lambda x: x * 255)
    normalize_transform = T.Normalize(
        mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]
        )

    transform = T.Compose([
            T.ToTensor(),
            to_bgr,
            to_255,
            normalize_transform
        ])

    return transform

def build_network_transform(minSize=800, maxSize=1333, mean=None, std=None):
    '''
    torchvision.models.detection apply a resize + normailze transform to input by default, since we already
    normalize before hand, it is important to disable it by setting mean/std to 0/1.

    Note: minSize/maxSize can however be set as wish, but should not be too far from default.
    '''
    if mean == None:
        mean = (0, 0, 0)
    if std == None:
        std = (1, 1, 1)

    transform = GeneralizedRCNNTransform(minSize, maxSize, mean, std)

    return transform

## Try to get it onto GPU:

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='/home/ammon/Documents/Scripts/keypoint_detection/models/keypoint_model_checkpoint.pt', help='Path to pretrained checkpoint')
parser.add_argument('--data_dir', default=None, required=True, help='Directory containing video')
parser.add_argument('--video', default=None, required=True, help='Video name')
parser.add_argument('--visualize', default=False, action='store_true', help='Save frames and visualize keypoints')
parser.add_argument('--calib_file', default=None, help='Camera calibration')

args = parser.parse_args()

transform = build_transform()
network_transform = build_network_transform()

backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=False)
model = MaskRCNN(backbone, num_classes=2)
model.to(device)
model.transform = network_transform
model.eval()
#model.load_state_dict(torch.load('/NAS/home/MaskRCNN_Torch_Bird/MaskRCNN_Torch_Bird/model_5.pth'))
model.load_state_dict(torch.load('/home/ammon/Documents/Scripts/keypoint_detection/MaskRCNN_Torch_Bird/model_5.pth'))

model_keypoints = pose_resnet(resnet_layers=50, num_classes=20).cuda()
model_keypoints.to(device)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model_keypoints.load_state_dict(checkpoint['model'])
model_keypoints.cuda()
model_keypoints.eval()
keypoint_transform_list = []
keypoint_transform_list.append(CropAndPad(out_size=(256, 256)))
keypoint_transform_list.append(T.ToTensor())
keypoint_transform_list.append(Normalize())
keypoint_transform = T.Compose(keypoint_transform_list)

out_dir = os.path.join(args.data_dir, args.video.split('.')[0])
img_dir = os.path.join(out_dir, 'images')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cap = cv2.VideoCapture(os.path.join(args.data_dir, args.video))
cnt = 0
keypoints_2d = []
keypoints_3d= []
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:, :, ::-1]
    frames = []
    frames_orig = []
    frames_keypoints = []
    height, width = frame.shape[:2]
    frames.append(transform(Image.fromarray(frame[:height//2, :width//2, :]).convert('RGB')))
    frames.append(transform(Image.fromarray(frame[:height//2, width//2:, :]).convert('RGB')))
    frames.append(transform(Image.fromarray(frame[height//2:, :width//2, :]).convert('RGB')))
    frames.append(transform(Image.fromarray(frame[height//2:, width//2:, :]).convert('RGB')))
    frames_orig.append(frame[:height//2, :width//2, :])
    frames_orig.append(frame[:height//2, width//2:, :])
    frames_orig.append(frame[height//2:, :width//2, :])
    frames_orig.append(frame[height//2:, width//2:, :])

    with torch.no_grad():
        #outputs = model(frames)
        outputs = model([f.cuda() for f in frames])
    # import ipdb
    # ipdb.set_trace()
    boxes = []
    offset = []
    scales = []
    orig_scales = []
    frame_num = str(cnt).zfill(6)
    for i in range(len(frames)):
        box = outputs[i]['boxes'][0].cpu().numpy()
        center_x = 0.5*(box[0] + box[2])
        center_y = 0.5*(box[1] + box[3])
        scale = max(box[2] - box[0], box[3] - box[1])
###NOTE: Need to get this scaling right to both catch the tale and note fale everything
        scale = int(1.2 * scale)
        #scale = 1.4 * scale
        scales.append(scale)
        min_x = max(center_x - 0.5 * scale, 0)
        min_y = max(center_y - 0.5 * scale, 0)
        max_x = min(center_x + 0.5 * scale, width * .5)
        max_y = min(center_y + 0.5 * scale, height * .5)
        orig_scales.append((int(max_y) - int(min_y),int(max_x)-int(min_x)))
        box = np.array([min_x, min_y, max_x, max_y]).astype(int)
        boxes.append(box)
        offset.append([min_x, min_y])
        box_keypoints = np.array([ [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y] ]).astype(int)
        frames_keypoints.append(keypoint_transform((Image.fromarray(frames_orig[i]), box_keypoints)))
    keypoint_frames = torch.stack(frames_keypoints, dim=0).cuda()
    scale = torch.tensor(scales).cuda().view(-1)
    offset = torch.tensor(offset).cuda().view(-1, 2)
    with torch.no_grad():
        output = model_keypoints(keypoint_frames)[0]
        keypoint_locs = torch.from_numpy(heatmaps_to_locs(output)).cuda()
        keypoint_locs[:,:,:-1] = keypoint_locs[:,:,:-1] * 4 * scale[:,None,None] / 256.
        keypoint_locs[:,:,:-1] += offset[:,None,:]
    heatmaps = np.zeros((len(frames), output.shape[1], 1024, 1024))
    for i in range(len(frames)):
        heatmaps_orig = F.interpolate(output[i].unsqueeze(0).cpu(), size=orig_scales[i], mode='bilinear')
        min_x = boxes[i][0]
        min_y = boxes[i][1]
        max_x = boxes[i][2]
        max_y = boxes[i][3]
        heatmaps[i, :, min_y:max_y, min_x:max_x] = heatmaps_orig
    round1 = voxel_keypoints3(heatmaps,args.calib_file,res=100)
    round2 = voxel_keypoints3(heatmaps,args.calib_file,res=20,grids=round1)
    round3 = voxel_keypoints3(heatmaps,args.calib_file,res=5,grids=round2)
    points,_ =  voxel_keypoints3(heatmaps,args.calib_file,res=1,grids=round3)
    keypoints_3d.append(points)
    #keypoints_3d.append(voxel_keypoints(heatmaps, args.calib_file))
    keypoint_locs = keypoint_locs.cpu().numpy()
    keypoints_2d.append(keypoint_locs)
    for i in range(4):
        I = frames_orig[i].astype(np.uint8)
        box = boxes[i]
        start = (box[0],box[1])
        end = (box[2], box[3])
        I = cv2.rectangle(I, start, end, [255,0,0],  thickness=5)
        keypoint = keypoint_locs[i, :, :-1]
        vkeypoint = {}
        for key in visualization_keypoints.keys():
            vkeypoint[key] = keypoint[visualization_keypoints[key], :2]
        if args.visualize:
            I = draw_skeleton(I, vkeypoint, radius=4)
            img_fname = os.path.join(img_dir, 'frame_'+frame_num+'_cam_'+str(i)+'.jpg')
            cv2.imwrite(img_fname, I[:, :, ::-1])
    cnt += 1
keypoints_2d = np.stack(keypoints_2d, axis=-1)
keypoints_3d = np.stack(keypoints_3d, axis=-1)
np.save(os.path.join(out_dir, 'pred_keypoints_2d.npy'), keypoints_2d)
np.save(os.path.join(out_dir,'pred_keypoints_3d.npy'),keypoints_3d)
# args.calib_file file should be the calibration.yaml file that Berndt generated
#compute_3d_pose(out_dir, calib_file=args.calib_file)
