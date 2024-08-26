import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time


# Load the mesh data from the '.obj' file.
def load_obj(file_path):
    vertices = []
    normals = []
    textures = []
    faces = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            if parts[1] == 'n':
                # Vertex normal information
                normals.append(list(map(float, parts[2:])))
            elif parts[1] == 't':
                # Texture vertex information
                textures.append(list(map(float, parts[2:])))
            else:
                # Vertex position information
                vertices.append(list(map(float, parts[1:])))
        elif parts[0] == 'f':
            # Face information
            face = []
            face = [int(x.split('/')[0])-1 for x in parts[1:]]
            faces.append(face)
            # for item in parts[1:]:
            #     vertex_data = item.split('/')
            #     vertex_index = int(vertex_data[0]) - 1
            #     texture_index = int(vertex_data[1]) - 1 if vertex_data[1] else 0
            #     normal_index = int(vertex_data[2]) -1 if vertex_data[2] else 0
            #     face.append((vertex_index, texture_index, normal_index))
            # faces.append(face)
    
    return np.array(vertices), np.array(normals), np.array(textures), np.array(faces)



def main():

    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    tool = 'open3d' #  open3d or plt
    read_view = False
    save_view = False
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_path = os.path.join(save_dir, clip, 'vis_mesh')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        mesh_path = os.path.join(path, 'meshes')
        mesh_files = sorted(glob(mesh_path + '/'+ '*.obj'))
        for mesh_file in tqdm(mesh_files[1:]):
            if tool == 'plt':
                mesh_vertices, mesh_normals, mesh_textures, mesh_faces = load_obj(mesh_file)
                # Create a 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Plot the mesh
                ax.plot_trisurf(mesh_vertices[:, 0], -mesh_vertices[:, 1], triangles=mesh_faces, Z=-mesh_vertices[:, 2], cmap='viridis')
                # Customize the plot
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                # Show the plot
                # plt.show()
                plt.savefig(os.path.join(vis_path, mesh_file.split('.')[-2]))
                plt.clf()
                plt.close()
            if tool == 'open3d':
                vis = o3d.visualization.Visualizer()
                vis.create_window(width=1280,height=720)
                mesh = o3d.io.read_triangle_mesh(os.path.normpath(mesh_file), True)
                vis.add_geometry(mesh)
                ctr = vis.get_view_control()
                if save_view:
                    vis.run()
                    param = ctr.convert_to_pinhole_camera_parameters()
                    o3d.io.write_pinhole_camera_parameters('/home/fangqiang/thermal-hand/view.json', param)
                    break
                if read_view:
                    param = o3d.io.read_pinhole_camera_parameters('/home/fangqiang/thermal-hand/view.json')
                    ctr.convert_from_pinhole_camera_parameters(param,allow_arbitrary=True)
                vis.poll_events()
                vis.update_renderer()
                # o3d.visualization.draw_geometries([mesh])
                vis.capture_screen_image(os.path.join(vis_path, mesh_file.split('.')[-2] + '.png'))
                vis.destroy_window()
            


if __name__ == "__main__":
    main()
