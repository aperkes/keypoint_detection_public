import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from models import pose_resnet
import argparse
from visualize_keypoints import visualization_keypoints, draw_skeleton
from compute_3d_pose import compute_3d_pose
from voxel_carving import voxel_carving_iterative
from voxel_carving import voxel_carving3

## Helper function to build model
def get_model_instance_segmentation(num_classes):
# load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Get number of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256 ##don't quite understand that...
# and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model

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

def build_mask_transform():
    transform = T.Compose([T.ToTensor()])
    return transform

## Try to get it onto GPU:

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='/home/ammon/Documents/Scripts/keypoint_detection/models/keypoint_model_checkpoint.pt', help='Path to pretrained checkpoint')
parser.add_argument('--data_dir', default='/data/birds/postures/birdview-2019/', required=False, help='Directory containing video')
parser.add_argument('--video', default='2019-06-08-13-16-12_BDY.wav.mp4', required=False, help='Video name')
parser.add_argument('--visualize', default=False, action='store_true', help='Save frames and visualize keypoints')
parser.add_argument('--pca',default=False,action='store_true',help = 'Save 1st and 2nd principal components')
parser.add_argument('--calib_file', default='/data/birds/postures/calibrations/birdview/2019-06-08-13-16-12_BDY/calibration/calibration.yaml', help='Camera calibration')
args = parser.parse_args()

transform = build_transform()
network_transform = build_mask_transform()
"""
backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=False)
model = MaskRCNN(backbone, num_classes=2)
"""

model_save = '/home/ammon/Documents/Scripts/bird_segment/model_9.pt'
model = get_model_instance_segmentation(2)
model.to(device)
#model.transform = network_transform
model.eval()
#model.load_state_dict(torch.load('/NAS/home/MaskRCNN_Torch_Bird/MaskRCNN_Torch_Bird/model_5.pth'))
#model.load_state_dict(torch.load('/home/ammon/Documents/Scripts/keypoint_detection/MaskRCNN_Torch_Bird/model_5.pth'))
model.load_state_dict(torch.load(model_save))

out_dir = os.path.join(args.data_dir, args.video.split('.')[0])
print('storing data to',out_dir)

"""#Comment out these lines if you are doing keypoints
model_keypoints = pose_resnet(resnet_layers=50, num_classes=20).cuda()
model_keypoints.to(device)

if args.checkpoint is not None:
    print('loading checkpoint')
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
"""# Uncomment

cap = cv2.VideoCapture(os.path.join(args.data_dir, args.video))
cnt = 0
keypoints_2d = []
volumes = []
clouds = []
count = 0
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:, :, ::-1]
    frames = []
    frames_orig = []
    frames_keypoints = []
    height, width = frame.shape[:2]
## Split out the mp4 frame into the 4 original camera frames (and convert to RGB)
    frames.append(network_transform(Image.fromarray(frame[:height//2, :width//2, :]).convert('RGB')))
    frames.append(network_transform(Image.fromarray(frame[:height//2, width//2:, :]).convert('RGB')))
    frames.append(network_transform(Image.fromarray(frame[height//2:, :width//2, :]).convert('RGB')))
    frames.append(network_transform(Image.fromarray(frame[height//2:, width//2:, :]).convert('RGB')))
    frames_orig.append(frame[:height//2, :width//2, :])
    frames_orig.append(frame[:height//2, width//2:, :])
    frames_orig.append(frame[height//2:, :width//2, :])
    frames_orig.append(frame[height//2:, width//2:, :])

    with torch.no_grad():
        #outputs = model(frames)
## Spit out the outputs of maskRCNN
        outputs = model([f.cuda() for f in frames])
    # import ipdb
    # ipdb.set_trace()
    boxes = []
    offset = []
    scales = []
    masks = []
    frame_num = str(cnt).zfill(6)
    for i in range(len(frames)):
        box = outputs[i]['boxes'][0].cpu().numpy()
## I'm not sure why I need the extra 0 here, maybe for ultiple instances of the same thing?
        mask = outputs[i]['masks'][0][0].cpu().numpy()
        masks.append(mask)
        center_x = 0.5*(box[0] + box[2])
        center_y = 0.5*(box[1] + box[3])
        scale = max(box[2] - box[0], box[3] - box[1])
###NOTE: Need to get this scaling right to both catch the tale and note fale everything
###NOTE: THIS IS WHERE I do voxel carving!! But I need to get the masks, not the boxes. outputs[i]['masks'][0]
## I think I can boot it up as a thread in python and it won't even slow me down. 
        scale = 1.2 * scale
        #scale = 1.4 * scale
        scales.append(scale)
        min_x = max(center_x - 0.5 * scale, 0)
        min_y = max(center_y - 0.5 * scale, 0)
        max_x = min(center_x + 0.5 * scale, width/2)
        max_y = min(center_y + 0.5 * scale, height/2)
        box = np.array([min_x, min_y, max_x, max_y]).astype(int)
        box_keypoints = np.array([ [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y] ]) 
        box = box.astype(int)
        boxes.append(box)
        offset.append([min_x, min_y])
# UNCOMMENT for keypoint detection
        #frames_keypoints.append(keypoint_transform((Image.fromarray(frames_orig[i]), box_keypoints)))
    count += 1
    masks = np.array(masks)
    #np.save(out_dir + '/mask_' + str(count),masks)
    #print('voxel_carving frame',count)
    #point_cloud,meta_data = voxel_carving(masks,args.calib_file,count=count,plot_me=True)
    #point_cloud,meta_data = voxel_carving_iterative(masks,args.calib_file,count=count)
    #clouds.append(np.array(point_cloud))
    #volumes.append(meta_data['n_points'])
    #print(meta_data['n_points'])
    #print('new iteration approach:')
    round1 = voxel_carving3(masks,args.calib_file,count=count,res=100)
    round2 = voxel_carving3(masks,args.calib_file,count=count,res=20,grids=round1)
    #round3 = voxel_carving3(masks,args.calib_file,count=count,res=5,grids=round2,)
    round4 = voxel_carving3(masks,args.calib_file,count=count,res=5,grids=round2,pca=args.pca,plot_me=args.visualize,out_dir=out_dir)
    print(args.video,'Frame:',count)
    #print('n-points:',len(round3[0]))
    volumes.append(len(round4[0]))
    clouds.append(round4[0])

    mask_name = out_dir + '/masks/masks_' + str(count) + '.npy'
    box_name = out_dir + '/masks/box_' + str(count) + '.npy'
    np.save(mask_name,masks)
    np.save(box_name,np.array(boxes))

    """Comment these out too if you're not doing keypoint
    keypoint_frames = torch.stack(frames_keypoints, dim=0).cuda()
    scale = torch.tensor(scales).cuda().view(-1)
    offset = torch.tensor(offset).cuda().view(-1, 2)

    with torch.no_grad():
        output = model_keypoints(keypoint_frames)[0]
        keypoint_locs = torch.from_numpy(heatmaps_to_locs(output)).cuda()
        keypoint_locs[:,:,:-1] = keypoint_locs[:,:,:-1] * 4 * scale[:,None,None] / 256.
        keypoint_locs[:,:,:-1] += offset[:,None,:]
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
np.save(os.path.join(out_dir, 'pred_keypoints_2d.npy'), keypoints_2d)
# args.calib_file file should be the calibration.yaml file that Berndt generated
compute_3d_pose(out_dir, calib_file=args.calib_file)
"""

cloud_file = args.video.split('.')[0] + '_cloud.npy'
volume_file = args.video.split('.')[0] + '_volume.npy'
np.save(out_dir + '/' + cloud_file,np.array(clouds))
np.save(out_dir + '/' + volume_file,np.array(volumes))
print('All done!')
