import os
import math

def find_maxmin_edges_in_obj_folder(obj_folder):
    max_edges = 0
    min_edges = 1000000

    for root, dirs, files in os.walk(obj_folder):
        for file in files:
            if file.endswith('.obj'):
                obj_path = os.path.join(root, file)
                file_edges = 0

                with open(obj_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('f '):
                            vertices = line[2:].split()
                            num_vertices = len(vertices)
                            file_edges += num_vertices

                max_edges = max(max_edges, file_edges)
                min_edges = min(min_edges, file_edges)

    return max_edges, min_edges

# Main #

obj_folder = 'datasets/testseeds/'
max_edges, min_edges = find_maxmin_edges_in_obj_folder(obj_folder)
max_edges = 750 * (math.ceil(max_edges/750))
min_edges = 750 * (math.ceil(min_edges/750))
print('max_edges: ', max_edges)
print('min_edges: ', min_edges)
pool4 = round(max_edges*0.5)
pool = round((max_edges - pool4)/5)
pool3 = pool4 + pool
pool2 = pool3 + pool
pool1 = pool2 + pool

print('max_faces: ', pool4)
print('ninput_edges: ', max_edges)
print('pool_res: ', pool1,pool2,pool3,pool4)
