import numpy as np
import pickle
seq = np.load("case2_male_custom_shape.pkl",allow_pickle=True)
#for key, data in seq:
#    output_dict[key] = data.detach().cpu().numpy().squeeze()
#print(output_dict)
print(seq)

seq2 = seq["body_pose"].detach().cpu().numpy().squeeze()

print(seq2)
output_dict = {}
output_dict['body_pose'] = seq2
print(output_dict)
#print(seq["joints"][0][0])
outjoint_path = './body_pose.pkl'
with open(outjoint_path, 'wb') as f:
                pickle.dump(output_dict, f)
#print(seq)


BATCH_SIZE = seq['poses'].shape[0]
smplm, smplf, _ = load_models(True, batch_size=BATCH_SIZE, model_type=args.model_type)
thetas = torch.from_numpy(seq['poses'][:BATCH_SIZE, 3:66]).float()
if MODEL_TYPE == 'smpl':
    thetas = torch.cat((thetas, torch.zeros(BATCH_SIZE, 1 * 3 * 2)), dim=1)
elif MODEL_TYPE in ['smplx', 'smplh']:
    thetas = torch.cat((thetas, torch.zeros(BATCH_SIZE, 1 * 3 * 0 + 1 * 3 * 0)), dim=1)
pose_hand = torch.from_numpy(seq['poses'][:BATCH_SIZE, 66:]).float()
global_orient = torch.from_numpy(seq['poses'][:BATCH_SIZE, :3]).float()
# get betas
betas = torch.from_numpy(seq['betas'][:args.num_betas]).float().unsqueeze(0).repeat(thetas.shape[0], 1)
# get root translation
trans = torch.from_numpy(seq['transl'][:BATCH_SIZE]).float()
subject_gender = seq['gender']
# SMPL forward with gender specific model
if str(seq['gender']).lower() == 'male':
    SMPLOutput = smplm.forward(transl=trans,
                               global_orient=global_orient,
                               hand_pose=pose_hand,
                               body_pose=thetas,
                               betas=betas,
                               pose2rot=True, )
    faces = smplm.faces
elif str(seq['gender']).lower() == 'female':
    SMPLOutput = smplf.forward(transl=trans,
                               global_orient=global_orient,
                               body_pose=thetas,
                               hand_pose=pose_hand,
                               betas=betas,
                               pose2rot=True, )
    faces = smplf.faces
else:
    SMPLOutput = smplm.forward(transl=trans,
                               global_orient=global_orient,
                               body_pose=thetas,
                               hand_pose=pose_hand,
                               betas=betas,
                               pose2rot=True, )
    faces = smplm.faces
all_step_vertices = SMPLOutput.vertices.to(DEVICE).float()  # torch.Size([100, 3])
print('SMPLOutput.vertices.shape: ', all_step_vertices.shape)

save_folder = '/'.join(sample_paths[seq_no].split('/')[3:-1])
obj_name = sample_paths[seq_no].split('/')[-1]
for i in range(all_step_vertices.shape[0]):
    if args.model_type in ['smpl', 'smplh']:
        mesh = trimesh.Trimesh(vertices=c2c(all_step_vertices[i]), faces=faces,
                               vertex_colors=np.tile(colors['grey'], (6890, 1)))
    else:
        mesh = trimesh.Trimesh(vertices=c2c(all_step_vertices[i]), faces=faces, vertex_colors=np.tile(colors['grey'], (10475, 1)))
    os.makedirs(osp.join(OUTPUT_DIR, save_folder), exist_ok=True)
    save_dir = osp.join(OUTPUT_DIR, save_folder, '{0}_{1:02d}.obj'.format(obj_name, i))
    save_mesh(mesh, None, save_dir)
    print('Saved to {}'.format(save_dir))
    obj_list.append(save_dir)