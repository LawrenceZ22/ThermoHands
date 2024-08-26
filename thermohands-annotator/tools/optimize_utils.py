import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def distort_coordinates(coords, camera_matrix, dist_coeffs):
    """
    Apply distortion to a grid of coordinates.
    coords should be an array of shape (N, 1, 2) and of type np.float32.
    """
    k1, k2, p1, p2, k3 = dist_coeffs
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    x = (coords[:, :, 0] - cx) / fx
    y = (coords[:, :, 1] - cy) / fy

    r = np.sqrt(x**2 + y**2)

    x_distorted = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_distorted = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    x_distorted = x_distorted * fx + cx
    y_distorted = y_distorted * fy + cy

    distorted_coords = np.concatenate((x_distorted[..., np.newaxis], y_distorted[..., np.newaxis]), axis=-1)

    return np.round(distorted_coords)

def align_depth2rgb(rgb_file, depth_file, rgb_camera_matrix, rgb_distortion_coeffs, \
                    depth_camera_matrix, depth_distortion_coeffs, rotation_matrix, translation_vector):
    
    # Capture depth images
    rgb_image = cv2.imread(rgb_file)
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    h, w = depth_image.shape[:2]
    
    # Undistort the depth image
    depth_img_undistorted  = cv2.undistort(depth_image, depth_camera_matrix, depth_distortion_coeffs[0])

    # Create a meshgrid of pixel indices
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)

    # Transform pixel indices to normalized depth camera coordinates
    X = (xv.flatten() - depth_camera_matrix[0, 2]) / depth_camera_matrix[0, 0]
    Y = (yv.flatten() - depth_camera_matrix[1, 2]) / depth_camera_matrix[1, 1]
    Z = depth_img_undistorted * 0.001  # Assume depth is in millimeters, convert to meters
    points = np.stack([X, Y, np.ones_like(X)], axis=-1) * Z.flatten()[:,np.newaxis]

    # Transform the point cloud to the RGB camera's coordinate system
    points_transformed = (rotation_matrix @ points.T + translation_vector * 0.001).T

    # Project the point cloud onto the RGB image plane
    uvs = points_transformed[:, :2] / points_transformed[:, 2:3]
    uvs[:, 0] = uvs[:, 0] * rgb_camera_matrix[0, 0] + rgb_camera_matrix[0, 2]
    uvs[:, 1] = uvs[:, 1] * rgb_camera_matrix[1, 1] + rgb_camera_matrix[1, 2]
    uvs = np.round(uvs).astype(int)

    # apply the distortion of RGB camera
    uvs = distort_coordinates(uvs[:, np.newaxis, :2], rgb_camera_matrix, rgb_distortion_coeffs[0]).reshape(-1,2).astype(int)

    # Create an aligned depth image
    aligned_depth = np.zeros_like(rgb_image[:, :, 0], dtype=np.float32)
    valid = (uvs[:,0] >= 0) & (uvs[:,0] < rgb_image.shape[1]) & (uvs[:,1] >= 0) & (uvs[:,1] < rgb_image.shape[0])
    aligned_depth[uvs[:,1][valid], uvs[:,0][valid]] = points_transformed[:, 2][valid]

    return aligned_depth

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def convert_to_3d(keypoints_2d, depths, camera_matrix):
    # Extracting camera parameters
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    # Vectorized computation
    X = ((keypoints_2d[:,0] - cx) * depths) / fx
    Y = ((keypoints_2d[:,1] - cy) * depths) / fy
    Z = depths

    # Combine X, Y, Z to get 3D keypoints
    keypoints_3d = np.vstack((X, Y, Z))

    return keypoints_3d

def draw_losses(losses, save_path):

    n_plots = len(losses)
    fig, axs = plt.subplots(int(n_plots/2), 2, figsize=(15, 20), sharex=True)
    # Check if there's only one subplot (avoid indexing error)
    if n_plots == 1:
        axs = [axs]
    axs = axs.reshape(n_plots)
    for ax, (loss_name, loss_values) in zip(axs, losses.items()):
        ax.plot(loss_values, label=loss_name)
        ax.set_title(loss_name)
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    # Set common labels
    plt.xlabel("Epoch")
    plt.tight_layout()  # Adjust layout to not overlap
    # plt.show()
    plt.savefig(save_path, dpi = 300)
    plt.close()
    plt.clf()

def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=False, save_path = None):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    ax.view_init(elev = 60)
    if save_path:
        for i in range(joints.shape[0]):
            ax.text(joints[i, 0], joints[i, 1], joints[i, 2], s = str(i), fontsize=12)
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    plt.clf()

def draw_3d_keypoints(vis_path, ego_pose_l_3d, ego_pose_r_3d):

    # Draw lines connecting hand landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (0, 9), (0, 13), (0, 17)
    ]
    conn_color = [
        'red', 'red', 'red', 'red',
        'green', 'green', 'green',
        'blue', 'blue', 'blue',
        'purple', 'purple', 'purple',
        'cyan', 'cyan', 'cyan',
        'green', 'blue', 'purple', 'cyan'
    ]

    # Creating a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot
    # ax.scatter(ego_pose_l_3d[:,0], ego_pose_l_3d[:,1], ego_pose_l_3d[:,2])
    # ax.scatter(ego_pose_r_3d[:,0], ego_pose_r_3d[:,1], ego_pose_r_3d[:,2])
    ego_pose_l_3d = ego_pose_l_3d/1000 
    ego_pose_r_3d = ego_pose_r_3d/1000
    for (start, end), col in zip(connections, conn_color):
        start_point = ego_pose_l_3d[start]
        end_point = ego_pose_l_3d[end]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col, linewidth = 2)
        start_point = ego_pose_r_3d[start]
        end_point = ego_pose_r_3d[end]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col, linewidth = 2)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim([-0.2, 0.3])
        ax.set_ylim([-0.2, 0.3])
        ax.set_zlim([0.1, 0.6])
        ax.view_init(azim=115)
    
    # save the plot
    plt.savefig(vis_path)
    plt.close()
    plt.clf()

def display_hand_2d(verts_l, verts_r, transform, cam_matrix, cam_dist, rgb_file, save_file):

    image = cv2.imread(rgb_file)
    image = cv2.undistort(image, cam_matrix.cpu().numpy(), cam_dist.cpu().numpy())
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure()
    plt.imshow(rgb_image)
    verts_l = (verts_l/1000).transpose(1,0).to(torch.float32)
    verts_r = (verts_r/1000).transpose(1,0).to(torch.float32)
    transform = transform.to(torch.float32)
    verts_l = transform[:3, :3] @ verts_l + transform[:3, 3].unsqueeze(1)
    verts_r = transform[:3, :3] @ verts_r + transform[:3, 3].unsqueeze(1)
    verts_l = verts_l[:2] / verts_l[2]
    verts_r = verts_r[:2] / verts_r[2]
    verts_l[0] = verts_l[0] * cam_matrix[0, 0] + cam_matrix[0, 2]
    verts_l[1] = verts_l[1] * cam_matrix[1, 1] + cam_matrix[1, 2]
    verts_r[0] = verts_r[0] * cam_matrix[0, 0] + cam_matrix[0, 2]
    verts_r[1] = verts_r[1] * cam_matrix[1, 1] + cam_matrix[1, 2]
    verts_l = torch.round(verts_l).int().cpu().detach().numpy()
    verts_r = torch.round(verts_r).int().cpu().detach().numpy()
    connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (0, 9), (0, 13), (0, 17)
        ]
    for connection in connections:
        point1 = verts_l[:,connection[0]]
        point2 = verts_l[:,connection[1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue', linewidth=1)
    for connection in connections:
        point1 = verts_r[:, connection[0]]
        point2 = verts_r[:, connection[1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', linewidth=1)
    plt.scatter(verts_l[0], verts_l[1], s = 0.5, color='blue', label='Hand Vertices - L')
    plt.scatter(verts_r[0], verts_r[1], s = 0.5, color='red', label='Hand Vertices - R')
    plt.axis('off')  # Turn off axis
    plt.savefig(save_file)
    plt.close()
    plt.clf()


    
def find_nearest(tensor1, tensor2):
    """
    Find the nearest points in tensor1 for each point in tensor2, and their distances, and vice versa

    :param tensor1: Mx2 PyTorch tensor
    :param tensor2: Nx2 PyTorch tensor
    :return: A tuple containing:
             - Indices of the nearest points in tensor1 for each point in tensor2
             - The corresponding minimum distances
    """
    # Expand tensor1 and tensor2 for vectorized subtraction
    tensor1_expanded = tensor1.unsqueeze(1)  # Shape: Mx1x2
    tensor2_expanded = tensor2.unsqueeze(0)  # Shape: 1xNx2

    # Calculate pairwise squared Euclidean distances
    distances = (tensor1_expanded - tensor2_expanded).pow(2).sum(2)

    # Find the index and the value of the minimum distance for each point in tensor2
    min_distances, min_distance_indices = distances.min(dim=0)
    # min_distances_rev, min_distance_rev = distances.min(dim=1)

    # Return indices and square root of min distances for actual Euclidean distances
    return min_distances.sqrt()

# def pose_prior_error(theta, theta_prior):

def mesh_surface_error(hand_pcd, vertices):

    vertices = (vertices[0]/1000)
    min_distances = find_nearest(hand_pcd, vertices)
    return min_distances.mean()

def joint_limit_error(theta, limit_max, limit_min):
    rot_len = theta.shape[1] - 45
    errors = torch.maximum(torch.zeros(45).to('cuda'), limit_min-theta[0, rot_len:]) + torch.maximum(theta[0, rot_len:] - limit_max, torch.zeros(45).to('cuda'))
    return errors.mean()


def joint_2d_error(ego_pose, exo_pose, joints, transform, ego_cam_matrix, exo_cam_matrix):
    # Project joints
    joints = (joints[0]/1000).transpose(1,0)
    joints_exo = transform[:3, :3] @ joints + transform[:3, 3].unsqueeze(1)
    joints = joints[:2] / joints[2]
    joints_exo = joints_exo[:2] / joints_exo[2]
    joints[0] = joints[0] * ego_cam_matrix[0, 0] + ego_cam_matrix[0, 2]
    joints[1] = joints[1] * ego_cam_matrix[1, 1] + ego_cam_matrix[1, 2]
    joints_exo[0] = joints_exo[0] * exo_cam_matrix[0, 0] + exo_cam_matrix[0, 2]
    joints_exo[1] = joints_exo[1] * exo_cam_matrix[1, 1] + exo_cam_matrix[1, 2]
    errors = torch.norm(joints[:2] - ego_pose.transpose(1,0), dim=0)
    errors_exo = torch.norm(joints_exo[:2] - exo_pose.transpose(1,0), dim=0)
    return 2 * errors.mean() + errors_exo.mean()
    # return 2 * errors.mean()

def reg_pose_error(theta):

    rot_len = theta.shape[1] - 45
    error = torch.norm(theta[0, rot_len:])

    return error

def reg_shape_error(beta):

    error = torch.norm(beta)

    return error

def joint_3d_error(pose, joints):

    joints = (joints[0]/1000)
    errors =  torch.norm(joints - pose, dim=1)

    return errors.mean()

# def mask_error(mask, vertices, cam_matrix):

#     mask_render = torch.zeros(mask.shape).cuda()
#     # Project vertices
#     vertices = (vertices[0]/1000).transpose(1,0)
#     # ones = torch.ones(vertices.shape[0], 1, device=vertices.device)
#     # vertices_homo = torch.cat([vertices, ones], dim=1)
#     vertices = vertices[:2] / vertices[2]
#     vertices[0] = vertices[0] * cam_matrix[0, 0] + cam_matrix[0, 2]
#     vertices[1] = vertices[1] * cam_matrix[1, 1] + cam_matrix[1, 2]
#     vertices = torch.round(vertices)
#     vertices[:,0] = torch.clamp(vertices[:,0], min=0, max=mask_render.shape[1]-1)
#     vertices[:,1] = torch.clamp(vertices[:,1], min=0, max=mask_render.shape[0]-1)
#     vertices = vertices.long()
#     mask_render[vertices[:,1], vertices[:,0]] = 1


#     return 0

# def create_mesh(vertices):
#     # Assuming the vertices are of shape (N, 3)
#     # Create a dummy array of faces, which is just a set of triangles (N-2, 3)
#     faces = torch.tensor([[i, i+1, i+2] for i in range(vertices.shape[0] - 2)], dtype=torch.int64)
    
#     # Create a dummy texture
#     verts_rgb = torch.ones_like(vertices)[None]  # (1, N, 3)
#     textures = TexturesVertex(verts_features=verts_rgb)

#     return Meshes(verts=[vertices], faces=[faces], textures=textures)

# def render_mesh(mesh):
#     # Initialize a camera.
#     R, T = look_at_view_transform(2.7, 0, 0)
#     cameras = OpenGLPerspectiveCameras(device="cuda:0", R=R, T=T)

#     # Define the settings for rasterization and shading
#     raster_settings = RasterizationSettings(
#         image_size=256,
#         blur_radius=0.0,
#         faces_per_pixel=1,
#     )

#     # Create a renderer
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#         shader=SoftPhongShader(device="cuda:0", cameras=cameras)
#     )

#     # Render the mesh
#     images = renderer(mesh)

#     return images[0, ..., :3]

def silhouette_error(ego_mask, exo_mask, vertices, transform, ego_cam_matrix, exo_cam_matrix):
    # Project vertices
    vertices = (vertices[0]/1000).transpose(1,0)
    vertices_exo = transform[:3, :3] @ vertices + transform[:3, 3].unsqueeze(1)
    vertices = vertices[:2] / vertices[2]
    vertices_exo = vertices_exo[:2] / vertices_exo[2]
    vertices[0] = vertices[0] * ego_cam_matrix[0, 0] + ego_cam_matrix[0, 2]
    vertices[1] = vertices[1] * ego_cam_matrix[1, 1] + ego_cam_matrix[1, 2]
    vertices_exo[0] = vertices_exo[0] * exo_cam_matrix[0, 0] + exo_cam_matrix[0, 2]
    vertices_exo[1] = vertices_exo[1] * exo_cam_matrix[1, 1] + exo_cam_matrix[1, 2]
    mask_indices = torch.nonzero(ego_mask, as_tuple=False) # y x
    mask_indices_exo = torch.nonzero(exo_mask, as_tuple=False) # y x
    mask_indices = mask_indices[:,[1,0]]
    mask_indices_exo = mask_indices_exo[:,[1,0]]
    min_dist  = find_nearest(mask_indices, vertices[:2].transpose(1,0))
    min_dist_exo = find_nearest(vertices_exo[:2].transpose(1,0), mask_indices_exo)

    return 2 * min_dist.mean() +  min_dist_exo.mean() 
    # return 2 * min_dist.mean()
    
