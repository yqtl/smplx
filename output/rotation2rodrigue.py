import numpy as np
import json
import pickle
from utils import (
    Tensor, batch_rodrigues, apply_deformation_transfer, batch_rot2aa)
#seq = np.load("case2_male_custom_shape.pkl",allow_pickle=True)
seq = np.load("HumanShape_smplx2smpl2smplx.pkl",allow_pickle=True)
#seq = np.load("EvoSkeleton.pkl",allow_pickle=True)
#print(seq)
for key, data in seq.items():
    #output_dict[key] = data.detach().cpu().numpy().squeeze()
    print(key, data)
#print(seq)
#print((seq["body_pose"].detach().cpu().numpy().squeeze()))
#seq2 = seq["body_pose"].detach().cpu().numpy().squeeze()
#print("body pose")
#print(seq2)
#print(seq["joints"])
#print("convert rotational matrix to rotation vector")
#full_pose=seq["full_pose"].detach().cpu().numpy().squeeze()
full_pose=seq["full_pose"]
print(full_pose)
#pose_theta = batch_rot2aa(full_pose)
pose_theta = batch_rot2aa(seq['full_pose'][0]).reshape(1, 55*3)
print(pose_theta)
pose = pose_theta.detach().cpu().numpy().squeeze()
print(pose)
pose_data = {
            "pose": pose.tolist(),
        }
print(pose_data)
#outputpath = './case2_male_pose.json'
outputpath = './HumanShape_smplx2smpl2splx_pose.json'
with open(outputpath, "w") as f:
    json.dump(pose_data, f)

"""outjoint_path = './pose.pkl'
with open(outjoint_path, 'wb') as f:
                pickle.dump(output_dict, f)
"""
"""param_dict[key] = batch_rodrigues(
                var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
"""