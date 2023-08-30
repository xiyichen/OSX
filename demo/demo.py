import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import smplx
import pickle

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
import glob
import copy
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()

# prepare input image
transform = transforms.ToTensor()
os.makedirs(args.output_folder, exist_ok=True)
img_paths = sorted(glob.glob('/root/sergey/images/*'))

# model_params = dict(model_path='/root/OSX/common/utils/human_model_files/',
#                     # joint_mapper=JointMapper(joint_maps),
#                     create_global_orient=True,
#                     create_body_pose=True,
#                     create_betas=True,
#                     model_type='smplx',
#                     use_pca=True,
#                     num_betas=10,
#                     num_pca_comps=12)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# body_model = smplx.create(gender='neutral', **model_params).to(device)

body_model = copy.deepcopy(smpl_x.layer['neutral']).cuda()

for i, img_path in enumerate(img_paths):
    original_img = load_img(img_path)
    original_img_height, original_img_width = original_img.shape[:2]

    # detect human bbox with yolov5s
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vis_img = original_img.copy()
    for num, indice in enumerate(indices):
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        # mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        # mesh = mesh[0]

        # save mesh
        # save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))

        # render mesh
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        
        shape = out['smplx_shape'].float().cuda().reshape(1, -1)
        body_pose = out['smplx_body_pose'].float().cuda().reshape(1, -1)
        root_pose = out['smplx_root_pose'].float().cuda().reshape(1, -1)
        rhand_pose = out['smplx_rhand_pose'].float().cuda().reshape(1, -1)
        lhand_pose = out['smplx_lhand_pose'].float().cuda().reshape(1, -1)
        jaw_pose = out['smplx_jaw_pose'].float().cuda().reshape(1, -1)
        trans = out['cam_trans'].float().cuda().reshape(1, -1)
        zero_pose = torch.zeros((1, 3)).float().cuda()
        expr = out['smplx_expr'].float().cuda().reshape(1, -1)
        output = body_model(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # vertices_cam = output.vertices[0] + trans
        K = np.eye(3)
        K[0][0] = focal[0]
        K[1][1] = focal[1]
        K[0][2] = princpt[0]
        K[1][2] = princpt[1]
        K = torch.from_numpy(K).float().cuda()
        d = {}
        d['global_orient'] = root_pose.detach().cpu().numpy()
        d['body_pose'] = body_pose.detach().cpu().numpy()
        d['left_hand_pose'] = lhand_pose.detach().cpu().numpy()
        d['right_hand_pose'] = rhand_pose.detach().cpu().numpy()
        d['jaw_pose'] = jaw_pose.detach().cpu().numpy()
        d['leye_pose'] = zero_pose.detach().cpu().numpy()
        d['reye_pose'] = zero_pose.detach().cpu().numpy()
        d['expression'] = expr.detach().cpu().numpy()
        d['betas'] = shape.detach().cpu().numpy()
        d['camera_rotation'] = np.eye(3)
        d['camera_translation'] = trans.detach().cpu().numpy()
        d['focal_length'] = focal[0]
        d['camera_center'] = np.array(princpt)
        with open('/root/sergey/smplx_results/' + img_path.split('/')[-1].split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(d, f)


        
        # vertices_cam = vertices_cam/vertices_cam[:, -1:]
        # vertices_projected = (K@vertices_cam.T).T[:,:2].detach().cpu().numpy()
        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # for idx_, loc in enumerate(vertices_projected):
        #     x = int(loc[0])
        #     y = int(loc[1])
        #     print(x, y)
        #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        # cv2.imwrite('./smpl_debug.png', img)
        # print(vertices_projected)
        # exit()
        # mesh = vertices_cam.detach().cpu().numpy()
        
        # vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})

    # save rendered image
    # cv2.imwrite(os.path.join(args.output_folder, f'{i}_render.jpg'), vis_img[:, :, ::-1])
