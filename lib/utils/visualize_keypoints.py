import numpy as np
import cv2
import os

visualization_keypoints = {
    0: 'beak',
    1: 'left_eye',
    2: 'right_eye',
    3: 'nape',
    4: 'neck',
    5: 'crown',
    6: 'back',
    7: 'left_wing',
    8: 'right_wing',
    9: 'breast',
    10: 'tail',
    11: 'left_leg',
    12: 'right_leg',
    13: 'tail_tip',
    14: 'empty1',
    15: 'empty2',
    16: 'empty3'
}

"""
visualization_keypoints = {'Beak Tip': 0,
                           'Keel': 1,
                           'Tailbone': 2,
                           'Tip of Tail': 3,
                           'Left Eye': 4,
                           'Left Shoulder': 5,
                           'Left Wing Tip': 8,
                           'Left Knee': 9,
                           'Left Ankle': 10,
                           'Left Heel': 11,
                           'Right Eye': 12,
                           'Right Shoulder': 13,
                           'Right Wing Tip': 16,
                           'Right Knee': 17,
                           'Right Ankle': 18,
                           'Right Heel': 19,
                          }
"""
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 17. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    """
    Orig bird keypoints
    0: Beak Tip
    1: Keel
    2: Tailbone
    3: Tip of Tail
    4: Left Eye
    5: Left Shoulder
    6: Left Elbow
    7: Left Wrist
    8: Left Wing Tip
    9: Left Knee
    10: Left Ankle
    11: Left Heel
    12: Right Eye
    13: Right Shoulder
    14: Right Elbow
    15: Right Wrist
    16: Right Wing Tip
    17: Right Knee
    18: Right Ankle
    19: Right Heel
    """

    parents = [-1, 4, 10, 2, 0, 4, 5, 5, 6, 7, 1, 10, 10, 11, 12, 13, 14] 
    # parents = [-1] * 17
    joints = [
          joints['Beak Tip'],
          joints['Keel'],
          joints['Tailbone'],
          joints['Tip of Tail'],
          .5 * (joints['Left Eye'] + joints['Right Eye']),
          .5 * (joints['Left Shoulder'] + joints['Right Shoulder']),
          joints['Left Shoulder'],
          joints['Right Shoulder'],
          joints['Left Wing Tip'],
          joints['Right Wing Tip'],
          .5 * (joints['Left Knee'] + joints['Right Knee']),
          joints['Left Knee'],
          joints['Right Knee'],
          joints['Left Ankle'],
          joints['Right Ankle'],
          joints['Left Heel'],
          joints['Right Heel'],
      ]
    joints = np.array(joints)
    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L  wing
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R wing
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    colors = {k: [int(vi) for vi in totuple(v)] for k,v in colors.items()}

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)
    jcolors = [
        'red', 'red', 'red', 'red', 'red', 'red',
        'purple', 'purple', 'purple', 'purple', 'purple', 'purple',
        'purple', 'purple', 'purple', 'purple', 'purple', 'purple'
    ]

    ecolors = {
        0: 'green',
        1: 'purple',
        2: 'purple',
        3: 'blue',
        4: 'white',
        5: 'purple',
        6: 'light_green',
        7: 'green',
        8: 'light_green',
        9: 'green',
        10: 'purple',
        11: 'light_red',
        12: 'red',
        13: 'light_red',
        14: 'red',
        15: 'light_red',
        16: 'red',
    }

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'], -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (int(point_pa[0]), int(point_pa[1])), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                from IPython.core.debugger import Pdb
                Pdb().set_trace()
            cv2.line(image, (int(point[0]), int(point[1])), (int(point_pa[0]), int(point_pa[1])),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image
