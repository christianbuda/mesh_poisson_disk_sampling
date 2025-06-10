import numpy as np
from tqdm import tqdm
from collections import deque
import trimesh
import networkx as nx
import pygeodesic.geodesic as geodesic

# utils
def flat_remeshing(vertices, faces, only_outliers = True):
    # increases number of triangles in the mesh
    # if only_outliers is True, it does so only for the faces that have abnormally long edges
    # (useful to reduce mesh geodesic distance and graph geodesic distance discrepancies)
    
    if only_outliers:
        outliers = get_outliers(vertices, faces)
        while(len(outliers)>0):
            for t in tqdm(outliers):
                vertices, faces = split_triangle(t, vertices, faces)
            outliers = get_outliers(vertices, faces)
    else:
        for t in tqdm(range(len(faces))):
                vertices, faces = split_triangle(t, vertices, faces)
    
    return(vertices, faces)

def get_outliers(vertices, faces):
    # returns the indices of the largest triangles
    
    # longest edge in each triangle
    max_lengths = edge_lengths(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).max(axis = -1)
    Q1, Q3 = np.quantile(max_lengths, [0.25, 0.75])
    IQR = Q3-Q1
    return np.nonzero(max_lengths > Q3+1.5*IQR)[0]

def edge_lengths(A,B,C):
    # returns the length of the three edges of the triangle defined by the input vertices
    return np.stack([np.linalg.norm(B-A, axis = -1),
                        np.linalg.norm(C-A, axis = -1),
                        np.linalg.norm(B-C, axis = -1)], axis = -1)
    

def split_triangle(triangle, vertices, faces):
    # splits triangle belonging to input mesh into four subtriangles
    
    A,B,C = faces[triangle]
    
    adjacent_faces = (np.sum(np.any([A,B,C] == faces[...,np.newaxis], axis = -1), axis = -1) == 2).nonzero()[0]
    adjAB = adjacent_faces[np.sum(np.any(faces[adjacent_faces,:,np.newaxis]==[A,B], axis = -1), axis = -1)==2]
    adjAC = adjacent_faces[np.sum(np.any(faces[adjacent_faces,:,np.newaxis]==[A,C], axis = -1), axis = -1)==2]
    adjBC = adjacent_faces[np.sum(np.any(faces[adjacent_faces,:,np.newaxis]==[B,C], axis = -1), axis = -1)==2]
    
    vAB = faces[adjAB][np.all(faces[adjAB,:, np.newaxis] != [A,B], axis = -1)][0]
    vAC = faces[adjAC][np.all(faces[adjAC,:, np.newaxis] != [A,C], axis = -1)][0]
    vBC = faces[adjBC][np.all(faces[adjBC,:, np.newaxis] != [B,C], axis = -1)][0]
    
    # compute the three new vertices
    cAB = vertices.shape[0]
    cBC = vertices.shape[0]+1
    cAC = vertices.shape[0]+2
    
    vertices = np.concatenate([vertices, midpoints_vertices(vertices[A], vertices[B], vertices[C])])
    
    # internal faces
    t1 = [A, cAB, cAC]
    t2 = [B, cBC, cAB]
    t3 = [C, cAC, cBC]
    
    # additional adjacent faces
    tv1 = [vAB, B, cAB]
    tv2 = [vAC, A, cAC]
    tv3 = [vBC, C, cBC]
    
    faces = np.concatenate([faces, np.stack([t1, t2, t3, tv1, tv2, tv3], axis = 0)])
    faces[triangle] = [cAB, cBC, cAC]
    faces[adjAB] = [A, vAB, cAB]
    faces[adjAC] = [C, vAC, cAC]
    faces[adjBC] = [B, vBC, cBC]
    
    return(vertices, faces)

def midpoints_vertices(A,B,C):
    # returns midpoints coordinates for the three vertices
    return np.stack([(A+B)/2, (B+C)/2, (A+C)/2], axis = 0)

def triangle_area(A,B,C):
    # area of triangle given coordinates of vertices A,B,C
    # A,B,C can be array of arrays of dimensions (N,3)
    return(np.linalg.norm(np.cross(B-A, C-A), axis = -1)/2)

def points_from_coeffs(vertices, faces, sampled_faces, sampled_coeff):
    # computes sampled points coordinates from trilinear local coordinates values
    return np.sum(vertices[faces[sampled_faces]]*np.broadcast_to(sampled_coeff[...,np.newaxis], (len(sampled_faces),3,3)), axis = 1)

def sample_mesh_points(vertices, faces, npoints, faces_to_sample = None, generator = None, return_sampled_faces = False, return_sampled_coeffs = False, return_face_tracker = False, face_tracker = None):
    # sample npoints uniformly on the input mesh and add vertices in those points
    # NOTE: this is done by sampling faces based on areas, and then by sampling uniformly inside the triangle
    # if faces_to_sample is not None, points are only sampled in the subset of faces specified
    # returns a new mesh in which the sampled vertices are the last npoints rows in vertices



    if generator is None:
        generator = np.random.default_rng()

    if faces_to_sample is None:
        faces_to_sample = np.arange(len(faces))
    
    A = vertices[faces[faces_to_sample,0]]
    B = vertices[faces[faces_to_sample,1]]
    C = vertices[faces[faces_to_sample,2]]

    p = triangle_area(A, B, C)
    p /= p.sum()
    sampled_faces = generator.choice(faces_to_sample, size = npoints, p = p, replace = True)

    beta, gamma = tuple(generator.uniform(size = (2, npoints)))

    # bring sampled points inside triangles
    beta[beta+gamma>1] = 1 - beta[beta+gamma>1]
    gamma[beta+gamma>1] = 1 - gamma[beta+gamma>1]
    alpha = 1-beta-gamma
    
    sampled_coeff = np.stack([alpha, beta, gamma], axis = -1)

    # ### OLD
    # A = vertices[faces[sampled_faces,0]]
    # B = vertices[faces[sampled_faces,1]]
    # C = vertices[faces[sampled_faces,2]]
    # sampled_points = alpha[:,np.newaxis]*A+beta[:,np.newaxis]*B+gamma[:,np.newaxis]*C
    
    sampled_points = points_from_coeffs(vertices, faces, sampled_faces, sampled_coeff)
    
    extra_outs = ()
    if return_sampled_faces:
        extra_outs += (sampled_faces,)
    if return_sampled_coeffs:
        extra_outs += (sampled_coeffs,)
    

    return add_points(vertices, faces, sampled_points, sampled_faces, face_tracker = face_tracker, return_face_tracker=return_face_tracker) + extra_outs

def get_submesh_faces(vertices, faces, source, radius, method = 'any'):
    # creates a submesh with all points at a distance 2*radius
    # and selects faces in the ring of distance [radius, 2*radius]
    
    if method == 'any':
        reducer = np.any
    elif method == 'all':
        reducer = np.all

    # select a subset of vertices using euclidean distance
    dists = np.linalg.norm(vertices-vertices[source], axis = -1)
    selected_vertices = np.asarray(dists<2*radius).nonzero()[0]
    
    # create euclidean submesh
    selected_faces = faces[reducer(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)]
    selected_faces = select_connected_faces(selected_faces, source)
    full_circle_vertices, full_circle_faces, selected_vertices = create_submesh(vertices, selected_faces)
    
    # keep track of source in the euclidean submesh space
    source = np.asarray(selected_vertices == source).nonzero()[0][0]
    
    # compute geodesic distances on the subset
    geoalg = geodesic.PyGeodesicAlgorithmExact(full_circle_vertices, full_circle_faces)
    dists = geoalg.geodesicDistances([source])[0]

    original_vertices = selected_vertices[dists<2*radius]   # vertices of the full circle submesh in the original mesh space

    # create full circle submesh
    selected_vertices = np.asarray(dists<2*radius).nonzero()[0]    # vertices of the full circle submesh in the euclidean submesh space
    selected_faces = full_circle_faces[reducer(np.any(selected_vertices == full_circle_faces[...,np.newaxis], axis = -1), axis = -1)]
    selected_faces = select_connected_faces(selected_faces, source)
    full_circle_vertices, full_circle_faces, selected_vertices = create_submesh(full_circle_vertices, selected_faces)
    
    # keep track of source in the full circle submesh space
    source = np.asarray(selected_vertices == source).nonzero()[0][0]
    
    dists = dists[selected_vertices]   # convert dists vector to full circle mesh space

    # vertices of the ring in the full circle submesh space
    selected_vertices = np.asarray(dists>radius).nonzero()[0]
    selected_faces = reducer(np.any(selected_vertices == full_circle_faces[...,np.newaxis], axis = -1), axis = -1).nonzero()[0]

    # faces of the full circle submesh in the original mesh space
    original_faces = reducer(np.any(original_vertices == faces[...,np.newaxis], axis = -1), axis = -1).nonzero()[0]

    return full_circle_vertices, full_circle_faces, source, selected_faces, original_faces

def select_connected_faces(faces, source, return_index = False):
    # NOTE: face_adjacency returns a list of indices of faces that are adjacent
    adj = trimesh.graph.face_adjacency(faces)
    
    # select a representative face
    # OLD: base_triangle = np.nonzero(faces == source)[0][0]
    # SLOWER: base_triangle = np.argmax(np.any(faces == source, axis = -1))
    base_triangle = np.argmax(faces == source)//3  # quickest
    
    graph = nx.Graph()
    graph.add_edges_from(adj)
    groups = nx.connected_components(graph)

    for g in groups:
        # if the connected component is the right one
        if base_triangle in g:
            if not return_index:
                return(faces[list(g)])
            else:
                return(faces[list(g)], list(g))
    
    raise BaseException('Connected component not found, something went horribly wrong...')


def create_submesh(vertices, faces):
    # removes superflous vertices not found in faces, and renumbers faces
    interesting_vertices = np.unique(faces)
    nverts = len(interesting_vertices)

    mask = (faces[...,np.newaxis] == interesting_vertices)

    newfaces = np.zeros(faces.shape + (nverts,), dtype = int)
    newfaces[mask] = np.broadcast_to(np.arange(nverts)[np.newaxis, np.newaxis], faces.shape + (nverts,))[mask]
    newfaces = newfaces.sum(axis = -1)
    
    return vertices[interesting_vertices], newfaces, interesting_vertices

def compute_dist_matrix(geoalg, vertices, verbose = False):
    # compute distance matrix between list of vertices
    # given PyGeodesicAlgorithmExact object in input
    
    nverts = len(vertices)
    out = np.zeros((nverts, nverts))
    
    # fill upper triangular part of matrix
    if verbose:
        iterator = tqdm(range(nverts-1))
    else:
        iterator = range(nverts-1)
        
    for idx in iterator:
        out[idx, idx+1:] = geoalg.geodesicDistances([vertices[idx]], vertices[idx+1:])[0]
    
    # simmetrize distance matrix
    i_lower = np.tril_indices(nverts, -1)
    out[i_lower] = out.T[i_lower]
    
    return out

def compute_close_vertices(vertices, faces, source, dist):
    # returns all vertices that are either closer than dist to the source vertex
    # or all vertices that are directly connected to one of the above ones
    
    # compute mesh edges (each edge is repeated twice)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    P1 = vertices[edges[:,0]]
    P2 = vertices[edges[:,1]]

    v = P2-P1

    P = vertices[source]

    # for numerical stability
    epsilon = np.std(v)**2*1e-8
    
    # optimal value to compute the distance between source and line spanned by the segment
    # i.e. the closest point to source that lies in the line spanned by P1 and P2 is  P1+t_min*(P2-P1)
    t_min = np.sum((P-P1)*v, axis = -1)/(np.sum(v**2, axis = -1)+epsilon)
    
    # clip the value to get a point inside the segment
    t_min = np.clip(t_min, a_min = 0, a_max = 1)

    # computes distance and selects the close ones
    selected_vertices = np.unique(edges[np.linalg.norm(t_min[:,np.newaxis]*v+P1-P, axis = -1)<dist])
    return selected_vertices

def compute_close_faces(vertices, faces, source, radius, fine_mesh = True, method = 'any', return_original_vertices = False):
    # creates a submesh with all points at a distance 2*radius
    # and selects faces in the ring of distance [radius, 2*radius]

    if method == 'any':
        reducer = np.any
    elif method == 'all':
        reducer = np.all

    # select a subset of vertices using euclidean distance
    if fine_mesh:
        dists = np.linalg.norm(vertices-vertices[source], axis = -1)
        selected_vertices = np.asarray(dists<2*radius).nonzero()[0]
    else:
        selected_vertices = compute_close_vertices(vertices, faces, source, 2*radius)
        
    # create euclidean submesh
    selected_faces_idx = reducer(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)
    selected_faces = faces[selected_faces_idx]
    original_faces = np.arange(len(faces))[selected_faces_idx]
    
    selected_faces, selected_faces_idx = select_connected_faces(selected_faces, source, return_index = True)
    original_faces = original_faces[selected_faces_idx]
    full_circle_vertices, full_circle_faces, selected_vertices = create_submesh(vertices, selected_faces)
    
    # keep track of source in the euclidean submesh space
    source = np.asarray(selected_vertices == source).nonzero()[0][0]
    original_vertices = selected_vertices.copy()   # vertices of the full circle submesh in the original mesh space
    
    # compute geodesic distances on the subset
    geoalg = geodesic.PyGeodesicAlgorithmExact(full_circle_vertices, full_circle_faces)
    dists = geoalg.geodesicDistances([source])[0]

    if fine_mesh:
        # WRONG?
        # original_vertices = selected_vertices[dists<2*radius]   # vertices of the full circle submesh in the original mesh space

        # create full circle submesh
        selected_vertices = np.asarray(dists<2*radius).nonzero()[0]    # vertices of the full circle submesh in the euclidean submesh space
        selected_faces_idx = reducer(np.any(selected_vertices == full_circle_faces[...,np.newaxis], axis = -1), axis = -1)
        selected_faces = full_circle_faces[selected_faces_idx]
        original_faces = original_faces[selected_faces_idx]
        
        selected_faces, selected_faces_idx = select_connected_faces(selected_faces, source, return_index = True)
        original_faces = original_faces[selected_faces_idx]
        full_circle_vertices, full_circle_faces, selected_vertices = create_submesh(full_circle_vertices, selected_faces)
        
        # keep track of source in the full circle submesh space
        source = np.asarray(selected_vertices == source).nonzero()[0][0]
        
        dists = dists[selected_vertices]   # convert dists vector to full circle mesh space
        original_vertices = original_vertices[selected_vertices]

    # vertices of the ring in the full circle submesh space
    selected_vertices = np.asarray(dists>radius).nonzero()[0]
    selected_faces = reducer(np.any(selected_vertices == full_circle_faces[...,np.newaxis], axis = -1), axis = -1).nonzero()[0]

    # faces of the full circle submesh in the original mesh space
    # original_faces = reducer(np.any(original_vertices == faces[...,np.newaxis], axis = -1), axis = -1).nonzero()[0]

    if not return_original_vertices:
        return full_circle_vertices, full_circle_faces, source, selected_faces, original_faces
    else:
        return full_circle_vertices, full_circle_faces, source, selected_faces, original_faces, original_vertices

class FaceTracker():
    def __init__(self, nfaces):
        self.tracker = np.arange(nfaces)
    
    def update_tracker(self, newfaces):
        self.tracker = self.tracker[newfaces]
    
    def find_faces(self, faces):
        return self.tracker[faces]


def poisson_disk_sampling(vertices, faces, min_dist = None, num_points = None, points_to_sample = None, seed_vertices = None, remesh = False, generator = None, return_original_faces = False, verbose = False):
    """
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        min_dist: float, minimum distance between the points of the poisson sampling
        num_points: (optional) int, rough number of points to sample, if min_dist is None, this should be given
        points_to_sample: points to sample in each disk
        seed_vertices: (optional) list of int, index of the vertices that should be included in the final sampling
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
                NOTE: sometimes it does not work, try upsampling with something more robust
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    # get a rough estimate of the number of points needed to cover the mesh
    total_area = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).sum()
    if min_dist is not None:
        num_points = int(0.5*total_area/min_dist**2)  # rough estimate, the 0.5 is empirical
        if verbose:
            print(f'Sampling about {num_points} points with a minimum distance of {min_dist}')
            
    elif min_dist is None:
        assert num_points is not None, 'If min_dist is None, num_points must be not None'
        min_dist = np.sqrt(0.5*total_area/num_points)
        
        if verbose:
            print(f'Sampling about {num_points} points with an estimated minimum distance of {min_dist}')
    
    if points_to_sample is None:
        # heuristics
        points_to_sample = int(30*np.max([(triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).max()-np.pi*min_dist**2)/(3*np.pi*min_dist**2), 1]))
    
    if verbose:
        print(f'Number of points sampled in each disk: {points_to_sample}')
    
    
    # variable that determines whether the mesh is fine enough for an accurate sampling
    fine_mesh = edge_lengths(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).mean()/min_dist
    if verbose:
        print(f'Mesh is{'' if fine_mesh<1 else ' not'} fine enough, with a coefficient of {fine_mesh}')
    # mesh is fine if fine_mesh<<1 (i.e. if a sphere of radius min_dist around a typical vertex contains many faces)
    fine_mesh = fine_mesh<1
    
    # face_tracker object
    nfaces = len(faces)
    face_tracker = FaceTracker(nfaces)
    
    Q = deque()

    if seed_vertices is None:
        # sample first point
        vertices, faces, sampled_dipoles, new_face_tracker, original_sampled_faces = sample_mesh_points(vertices, faces, npoints = 1, generator = generator, return_sampled_faces = True, return_face_tracker = True)
        sampled_dipoles = sampled_dipoles.tolist()
        original_sampled_faces = original_sampled_faces.tolist()
        face_tracker.update_tracker(new_face_tracker.tracker)
        
        Q.append(sampled_dipoles[0])
        
        if verbose:
            print(f'Starting sampling with vertex: {sampled_dipoles[0]}, with coordinates {vertices[sampled_dipoles[0]]}')
    else:
        if not isinstance(seed_vertices, np.ndarray):
            assert isinstance(seed_vertices, list), 'Seed vertices must be a list of mesh vertices'
        else:
            assert len(seed_vertices.shape) == 1 and seed_vertices.dtype == int, 'Seed vertices must be a list of mesh vertices'
        sampled_dipoles = []
        original_sampled_faces = []
        for v in seed_vertices:
            sampled_dipoles.append(v)
            original_sampled_faces.append(np.nan)
            Q.append(v)
        
        if verbose:
            print(f'Starting sampling with {len(seed_vertices)} seed vertices: {seed_vertices}, with coordinates {vertices[seed_vertices]}')


    iter = 0
    pbar = tqdm(total=num_points, desc="Poisson disk sampling points")
    while len(Q) > 0:
        pbar.set_description("Iteration %d" % (iter+1))
        # print(f'Iter: {iter}, dipole: {len(sampled_dipoles)}/{num_points}')
        
        # current point
        point = Q.popleft()
        
        # create circular submesh from which to sample
        full_circle_vertices, full_circle_faces, source, selected_faces, original_faces = compute_close_faces(vertices, faces, point, min_dist, fine_mesh = fine_mesh, method = 'any')
        
        # sample points
        full_circle_vertices, full_circle_faces, good_points, sampled_faces = sample_mesh_points(full_circle_vertices, full_circle_faces, faces_to_sample = selected_faces, npoints = points_to_sample, generator = generator, return_sampled_faces = True)
        
        # convert sampled_faces to main mesh space
        sampled_faces = original_faces[sampled_faces]
        
        # check if points are at the correct distance from current center
        geoalg = geodesic.PyGeodesicAlgorithmExact(full_circle_vertices, full_circle_faces)
        dists = geoalg.geodesicDistances([source], good_points)[0]
        good_points = good_points[(dists>min_dist)&(dists<2*min_dist)]
        sampled_faces = sampled_faces[(dists>min_dist)&(dists<2*min_dist)]

        if len(good_points)>0:
            # check which points are at the correct distance from each other
            dist_matrix = compute_dist_matrix(geoalg, good_points)
            tokeep = [0]
            for i in range(1,len(good_points)):
                if dist_matrix[tokeep, i].min()>min_dist:
                    tokeep.append(i)
            good_points = good_points[tokeep]
            sampled_faces = sampled_faces[tokeep]
        
        # add the good points so far to the main mesh
        vertices, faces, good_points, new_face_tracker = add_points(vertices, faces, full_circle_vertices[good_points], sampled_faces, return_face_tracker=True)
        
        ##### check if points are good wrt to main mesh

        # create submesh with radius 3*min_dist around current point to speed up computations
        # select a subset of vertices using euclidean distance
        dists = np.linalg.norm(vertices-vertices[point], axis = -1)
        selected_vertices = np.asarray(dists<3*min_dist).nonzero()[0]
        
        # create euclidean submesh
        selected_faces = faces[np.any(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)]
        selected_faces = select_connected_faces(selected_faces, point)
        submesh_vertices, submesh_faces, selected_vertices = create_submesh(vertices, selected_faces)
        
        # find already sampled points in submesh space
        _, points_indices, _ = np.intersect1d(selected_vertices, sampled_dipoles, assume_unique=True, return_indices=True)    
        old_points_to_check = np.arange(len(selected_vertices))[points_indices]
        
        # find points to test in submesh space (the difference with above is due to the fact that I already know that good_points is inside selected_vertices)
        points_indices = np.nonzero( selected_vertices == np.array(good_points)[:,np.newaxis] )[1]
        new_points_to_check = np.arange(len(selected_vertices), dtype = np.int32)[points_indices]
        
        # compute geodesic distances on the subset
        geoalg = geodesic.PyGeodesicAlgorithmExact(submesh_vertices, submesh_faces)
        tokeep = []
        tokeep_facetracker = []
        for i,p in enumerate(new_points_to_check):
            dists = geoalg.geodesicDistances([p], old_points_to_check)[0]
            if dists.min()>min_dist:
                tokeep.append(good_points[i])
                tokeep_facetracker.append(sampled_faces[i])
                Q.append(good_points[i])
        sampled_dipoles += tokeep
        original_sampled_faces += face_tracker.find_faces(tokeep_facetracker).tolist()
        
        # update face tracker object
        face_tracker.update_tracker(new_face_tracker.tracker)
        
        ###################################
        
        # OLD
        # if len(tokeep)>=points_to_sample-max_rejections:
        #     raise ValueError('Too few rejections!!!')
        
        # update progress bar
        iter += 1
        pbar.update(len(tokeep))
    pbar.close()
    
    if return_original_faces:
        return vertices, faces, sampled_dipoles, original_sampled_faces
    else:
        return vertices, faces, sampled_dipoles


def uniform_sampling(vertices, faces, num_points, remesh = False, return_original_faces = False, generator = None):
    """
    wrapper for sample_mesh_points()
    
    this uniform sampling is approximate, as each face is only selected once for the sampling, it's good in the limit where num_points << n_faces
    
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        num_points: int, number of points to sample
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    vertices, faces, sampled_points, sampled_faces =  sample_mesh_points(vertices, faces, npoints = num_points, generator = generator, return_sampled_faces = True)
    
    if return_original_faces:
        return vertices, faces, sampled_points, sampled_faces
    else:
        return vertices, faces, sampled_points


def distance_graph(vertices, faces):
    # create a weigthed graph from the mesh to allow edge path length computations
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # edges without duplication
    edges = mesh.edges_unique

    # create the corresponding graph to compute shortest line path
    G = nx.Graph()
    for edge in edges:
        weight = np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])
        G.add_edge(*edge, weight = weight)
    
    return G


def edge_distances(G, sources, targets):
    # computes edge distances between sources and targets
    dists = np.zeros((len(sources), len(targets)))
    
    all_dists = dict(nx.all_pairs_bellman_ford_path_length(G))
    
    for i in range(len(sources)):
        for j in range(len(targets)):
            dists[i,j] = all_dists[int(sources[i])][int(targets[j])]
    return dists

def compute_graph_dist_matrix(G, vertices):
    # compute distance matrix between list of vertices
    
    nverts = len(vertices)
    out = np.zeros((nverts, nverts))
    
    all_dists = dict(nx.all_pairs_bellman_ford_path_length(G))
    
    # fill upper triangular part of matrix
    for idx in range(nverts-1):
        for j in range(idx+1,nverts):
            out[idx,j] = all_dists[vertices[idx]][vertices[j]]
    
    # simmetrize distance matrix
    i_lower = np.tril_indices(nverts, -1)
    out[i_lower] = out.T[i_lower]
    
    return out


def edge_distance_poisson_disk_sampling(vertices, faces, min_dist = None, num_points = None, points_to_sample = None, seed_vertices = None, remesh = False, generator = None, return_original_faces = False, verbose = False):
    """
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        min_dist: float, minimum distance between the points of the poisson sampling
        num_points: (optional) int, rough number of points to sample, if min_dist is None, this should be given
        points_to_sample: points to sample in each disk
        seed_vertices: (optional) list of int, index of the vertices that should be included in the final sampling
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
                NOTE: sometimes it does not work, try upsampling with something more robust
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    # get a rough estimate of the number of points needed to cover the mesh
    total_area = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).sum()
    if min_dist is not None:
        num_points = int(0.5*total_area/min_dist**2)  # rough estimate, the 0.5 is empirical
        if verbose:
            print(f'Sampling about {num_points} points with a minimum distance of {min_dist}')
            
    elif min_dist is None:
        assert num_points is not None, 'If min_dist is None, num_points must be not None'
        min_dist = np.sqrt(0.5*total_area/num_points)
        
        if verbose:
            print(f'Sampling about {num_points} points with an estimated minimum distance of {min_dist}')
    
    
    if points_to_sample is None:
        # heuristics
        points_to_sample = int(30*np.max([(triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).max()-np.pi*min_dist**2)/(3*np.pi*min_dist**2), 1]))
    
    if verbose:
        print(f'Number of points sampled in each disk: {points_to_sample}')
    
    
    # variable that determines whether the mesh is fine enough for an accurate sampling
    fine_mesh = edge_lengths(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).mean()/min_dist
    if verbose:
        print(f'Mesh is{'' if fine_mesh<1 else ' not'} fine enough, with a coefficient of {fine_mesh}')
    # mesh is fine if fine_mesh<<1 (i.e. if a sphere of radius min_dist around a typical vertex contains many faces)
    fine_mesh = fine_mesh<1
    
    # face_tracker object
    nfaces = len(faces)
    face_tracker = FaceTracker(nfaces)
    
    Q = deque()

    if seed_vertices is None:
        # sample first point
        vertices, faces, sampled_dipoles, new_face_tracker, original_sampled_faces = sample_mesh_points(vertices, faces, npoints = 1, generator = generator, return_sampled_faces = True, return_face_tracker = True)
        sampled_dipoles = sampled_dipoles.tolist()
        original_sampled_faces = original_sampled_faces.tolist()
        face_tracker.update_tracker(new_face_tracker.tracker)
        
        Q.append(sampled_dipoles[0])
        
        if verbose:
            print(f'Starting sampling with vertex: {sampled_dipoles[0]}, with coordinates {vertices[sampled_dipoles[0]]}')
    else:
        if not isinstance(seed_vertices, np.ndarray):
            assert isinstance(seed_vertices, list), 'Seed vertices must be a list of mesh vertices'
        else:
            assert len(seed_vertices.shape) == 1 and seed_vertices.dtype == int, 'Seed vertices must be a list of mesh vertices'
        sampled_dipoles = []
        original_sampled_faces = []
        for v in seed_vertices:
            sampled_dipoles.append(v)
            original_sampled_faces.append(np.nan)
            Q.append(v)
        
        if verbose:
            print(f'Starting sampling with {len(seed_vertices)} seed vertices: {seed_vertices}, with coordinates {vertices[seed_vertices]}')


    iter = 0
    pbar = tqdm(total=num_points, desc="Poisson disk sampling points")
    while len(Q) > 0:
        pbar.set_description("Iteration %d" % (iter+1))
        # print(f'Iter: {iter}, dipole: {len(sampled_dipoles)}/{num_points}')
        
        # current point
        point = Q.popleft()
        
        # create circular submesh from which to sample
        full_circle_vertices, full_circle_faces, source, selected_faces, original_faces = compute_close_faces(vertices, faces, point, min_dist, method = 'any')        
        
        # sample points
        full_circle_vertices, full_circle_faces, good_points, sampled_faces = sample_mesh_points(full_circle_vertices, full_circle_faces, faces_to_sample = selected_faces, npoints = points_to_sample, generator = generator, return_sampled_faces = True)
        
        # convert sampled_faces to main mesh space
        sampled_faces = original_faces[sampled_faces]
        
        # check if points are at the correct distance from current center
        geoalg = distance_graph(full_circle_vertices, full_circle_faces)
        dists = edge_distances(geoalg, [source], good_points)[0]
        good_points = good_points[(dists>min_dist)&(dists<2*min_dist)]
        sampled_faces = sampled_faces[(dists>min_dist)&(dists<2*min_dist)]
        
        if len(good_points)>0:
            # check which points are at the correct distance from each other
            dist_matrix = compute_graph_dist_matrix(geoalg, good_points)
            tokeep = [0]
            for i in range(1,len(good_points)):
                if dist_matrix[tokeep, i].min()>min_dist:
                    tokeep.append(i)
            good_points = good_points[tokeep]
            sampled_faces = sampled_faces[tokeep]
        
        # add the good points so far to the main mesh
        vertices, faces, good_points, new_face_tracker = add_points(vertices, faces, full_circle_vertices[good_points], sampled_faces, return_face_tracker=True)
        
        ##### check if points are good wrt to main mesh

        # create submesh with radius 3*min_dist around current point to speed up computations
        # select a subset of vertices using euclidean distance
        dists = np.linalg.norm(vertices-vertices[point], axis = -1)
        selected_vertices = np.asarray(dists<3*min_dist).nonzero()[0]
        
        # create euclidean submesh
        selected_faces = faces[np.any(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)]
        selected_faces = select_connected_faces(selected_faces, point)
        submesh_vertices, submesh_faces, selected_vertices = create_submesh(vertices, selected_faces)
        
        # find already sampled points in submesh space
        _, points_indices, _ = np.intersect1d(selected_vertices, sampled_dipoles, assume_unique=True, return_indices=True)    
        old_points_to_check = np.arange(len(selected_vertices))[points_indices]
        
        # find points to test in submesh space (the difference with above is due to the fact that I already know that good_points is inside selected_vertices)
        points_indices = np.nonzero( selected_vertices == np.array(good_points)[:,np.newaxis] )[1]
        new_points_to_check = np.arange(len(selected_vertices), dtype = np.int32)[points_indices]
        
        # compute geodesic distances on the subset
        geoalg = distance_graph(submesh_vertices, submesh_faces)
        tokeep = []
        tokeep_facetracker = []
        for i,p in enumerate(new_points_to_check):
            dists = edge_distances(geoalg, [p], old_points_to_check)[0]
            if dists.min()>min_dist:
                tokeep.append(good_points[i])
                tokeep_facetracker.append(sampled_faces[i])
                Q.append(good_points[i])
        sampled_dipoles += tokeep
        original_sampled_faces += face_tracker.find_faces(tokeep_facetracker).tolist()
        
        # update face tracker object
        face_tracker.update_tracker(new_face_tracker.tracker)
        
        ###################################
        
        # OLD
        # if len(tokeep)>=points_to_sample-max_rejections:
        #     raise ValueError('Too few rejections!!!')
        
        # update progress bar
        iter += 1
        pbar.update(len(tokeep))
    pbar.close()
    
    if return_original_faces:
        return vertices, faces, sampled_dipoles, original_sampled_faces
    else:
        return vertices, faces, sampled_dipoles



##################### add points utils
def project_pointcloud_on_faces(points, vertices, faces):
    # this function provides the projection of the input point cloud on each triangle of the input mesh
    # it is unnecessarily convoluted to allow the use of the autograd package to compute gradients

    # (N_faces, N_vertex_per_face, 3)
    vertices_groups = vertices[faces]
    N_faces = faces.shape[0]
    N_points = points.shape[0]

    A = vertices_groups[:,0]
    B = vertices_groups[:,1]
    C = vertices_groups[:,2]

    R = create_rotation_matrices(np.cross(B-A, C-A))[np.newaxis]

    # add starting dimension to allow easier broadcasting for point cloud
    vertices_groups = np.broadcast_to(vertices_groups, (N_points,N_faces,3,3))
    A = A[np.newaxis]
    B = B[np.newaxis]
    C = C[np.newaxis]

    # this projects point P on the plane spanned by each triangle
    coeffs = np.linalg.inv(R[:,:,:2]@np.stack([B-A, C-A], axis = -1))@((R@(points[:,np.newaxis]-A)[...,np.newaxis])[:,:,:2])

    # coefficients of the trilinear coordinates that make up the projection on the triangle
    coeffs = np.array([[[[1], [0], [0]]]]) + np.array([[[[-1, -1], [1,0], [0,1]]]])@coeffs

    # # these are the actual projected points on the planes, the formula above is to find directly the trilinear coordinates
    # proj_P = np.squeeze(np.linalg.inv(R)@np.array([[[[1,0,0], [0,1,0], [0,0,0]]]])@R@(points[:,np.newaxis]-A)[..., np.newaxis])+A
    proj_P = np.sum(coeffs*vertices_groups, axis = 2)

    # check how many have exactly one negative coefficient
    pos_coeffs = coeffs[...,0]>0
    which_to_project = np.sum(pos_coeffs, axis = -1) == 2

    # this is a mask on coeffs that is equal to pos_coeffs in points that need to be projected, and is equal to [True, True, False] for all other points
    # it's a trick necessary to avoid item assignment as it is incompatible with autograd
    segments_endpoint = np.where(which_to_project[...,np.newaxis], pos_coeffs, np.ones(pos_coeffs.shape, dtype = bool)*np.array([[[1,1,0]]], dtype = bool))

    # once we have the mask, we can extract the indices
    index_points, index_faces, index_ending = np.nonzero(segments_endpoint)
    index_points = index_points[::2]
    index_faces = index_faces[::2]
    index_starting = index_ending[::2]
    index_ending = index_ending[1::2]

    # and the segments; keep in mind that these segments are only meaningful for the points that need line projection!
    line = vertices_groups[index_points, index_faces, index_ending].reshape((N_points,N_faces,3)) - vertices_groups[index_points, index_faces, index_starting].reshape((N_points,N_faces,3))

    # Use these masks as a trick to avoid item assignment
    # startpoints mask is True on the starting point of each segment (as defined in segments_endpoint)
    startpoints_mask = np.zeros((coeffs.shape))
    startpoints_mask[index_points, index_faces, index_starting] = 1
    # endpoints mask is True on the ending point of each segment (as defined in segments_endpoint)
    endpoints_mask = np.zeros((coeffs.shape))
    endpoints_mask[index_points, index_faces, index_ending] = 1

    # this is the coefficient of the new projected point, relative to the starting vertex
    # the coefficient relative to the ending vertex is its complement to 1 (i.e. 1-startingcoeff)
    tmp = (np.sum((proj_P-vertices_groups[index_points, index_faces, index_starting].reshape((N_points,N_faces,3)))*line, axis = -1)/np.linalg.norm(line, axis = -1)**2)[:,:,np.newaxis, np.newaxis]
    coeffs = (endpoints_mask*tmp + startpoints_mask*(1-tmp))*which_to_project[..., np.newaxis, np.newaxis] + coeffs*np.logical_not(which_to_project)[..., np.newaxis, np.newaxis]

    coeffs = np.clip(coeffs, 0, 1)
    coeffs = coeffs/np.sum(coeffs, axis = -2)[...,np.newaxis]

    proj_P = np.sum(coeffs*vertices_groups, axis = 2)

    return proj_P


def create_rotation_matrices(v, target = 'z'):
    # create a set of rotation matrices in 3D space in such a way that the list of vectors v are each rotated along the target direction
    # multidimensional version of "create_rotation_matrix"
    
    v = np.array(v, dtype=float)
    
    v_norm = np.linalg.norm(v, axis = 1)
    
    if np.any(v_norm == 0):
        raise ValueError("Some vectors in the array are null and cannot be rotated.")
    
    v = v / v_norm[...,np.newaxis]  # Normalize v
    
    if isinstance(target, str):
        if target == 'z':
            target = np.array([0, 0, 1])  # z-axis
        elif target == 'y':
            target = np.array([0, 1, 0])  # z-axis
        elif target == 'x':
            target = np.array([1, 0, 0])  # z-axis
        else:
            raise ValueError("target must be either a vector or one of ['x', 'y', 'z']")
    else:
        target = np.array(target, dtype=float)
    
        target_norm = np.linalg.norm(target)
        
        if np.isclose(target_norm, 0):
            raise ValueError("Zero vector cannot be a target.")
        
        target = target / target_norm  # Normalize target
        
    R = np.zeros((v.shape[0], 3, 3))
    
    
    # Compute rotation axes (cross product of v and target)
    axes = np.cross(v, target)
    axes_norm = np.linalg.norm(axes, axis = 1)
    axes /= axes_norm[...,np.newaxis]  # Normalize rotation axis
    
    # Compute rotation angle
    thetas = np.arccos(np.dot(v, target))[...,np.newaxis, np.newaxis]
    
    # as described in https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    R = axes[:, 2, np.newaxis, np.newaxis] * np.array([[[0,-1,0], [1,0,0], [0,0,0]]]) + axes[:, 1, np.newaxis, np.newaxis] * np.array([[[0,0,1], [0,0,0], [-1,0,0]]]) + axes[:, 0, np.newaxis, np.newaxis] * np.array([[[0,0,0], [0,0,-1], [0,1,0]]])
    
    R = np.eye(3)[np.newaxis] + np.sin(thetas) * R + (1 - np.cos(thetas)) * np.matmul(R,R)
    
    return R

def add_internal_points(vertices, faces, sampled_points, sampled_faces, face_tracker = None):
    # add sampled points and faces to the mesh
    # make sure points are internal to the faces (and not on the boundary)
    # generally faster than add_points()
    
    if len(sampled_points)>0:
        # NOTE: this function can only add one point per triangle at a time!
        if np.any(np.unique(sampled_faces, return_counts = True)[1]>1):
            raise ValueError('add_internal_points() only works when there is a single point per triangle to be added, found more than once here. Try refining the mesh or use add_points().')
        
        sampled_points_idx = np.arange(vertices.shape[0], vertices.shape[0] + sampled_points.shape[0])
        vertices = np.concatenate([vertices, sampled_points])
        
        # add the faces
        # B C D
        t1 = np.concatenate([faces[sampled_faces,1:], sampled_points_idx[:,np.newaxis]], axis = -1)
        # A D C
        t2 = np.concatenate([faces[sampled_faces,:1], sampled_points_idx[:,np.newaxis], faces[sampled_faces,2:]], axis = -1)
        
        faces = np.concatenate([faces, t1, t2])
        
        # replace first face
        faces[sampled_faces,2] = sampled_points_idx
        
        if face_tracker is not None:
            current_tracker = np.arange(len(faces))
            current_tracker[sampled_faces] = sampled_faces
            current_tracker[-2*len(sampled_faces):] = np.tile(sampled_faces, 2)
            face_tracker.update_tracker(current_tracker)
    else:
        sampled_points_idx = []
    
    return vertices, faces, sampled_points_idx
        

def find_close_points(vertices, points):
    # checks if any point in points is very close to a vertex
    
    rtol = 1e-5
    which_close = np.isclose(vertices[np.newaxis],points[:,np.newaxis], rtol = rtol).all(axis = -1)
    
    iter = 0
    while np.any(np.count_nonzero(which_close, axis =-1)>1):
        iter += 1
        
        rtol *= 0.9
        which_close = np.isclose(vertices[np.newaxis],points[:,np.newaxis], rtol = rtol).all(axis = -1)
        
        if iter>1000:
            raise BaseException('There may be a double vertex, check the mesh!')
    
    return which_close

def collinear(A,B,C, epsilon = 1e-2, return_value = False):
    # checks if the 3D points A, B, C are collinear
    # it does so by testing whether the distance between the middle point
    # and the segment between the other two is less than epsilon times smaller than the length of the segment itself
    
    # A, B, C can be arrays of N points, of dimension (N,3)
    
    longest_edge = edge_lengths(A,B,C).max(axis = -1)
    out = np.linalg.norm(np.cross(A-B, A-C), axis = -1)/longest_edge**2
    if not return_value:
        out = out<epsilon/2
    return out


def _add_collinear_point(A,B,C, vertices, faces, sampled_point, sampled_face, face_tracker = None):
    # adds a point to face [A B C] by splitting edge [A B] a placing the sampled point in the middle
    
    D = vertices.shape[0]
    vertices = np.concatenate([vertices, [sampled_point]])
    
    # new faces (where E is the vertex of the second face to which the edge [A B] belongs)
    # A D C, D B C, A E D, B D E
    
    
    second_face = np.nonzero((np.sum(np.any(faces[np.newaxis] == np.array([A,B])[...,np.newaxis, np.newaxis], axis = 0), axis = -1) == 2)&np.logical_not(np.any(faces == C, axis = -1)))[0]
    if len(second_face)>0:
        second_face = second_face[0]
        E = np.setdiff1d(faces[second_face], np.array([A,B]))[0]
        
        # D B C
        t1 = np.array([[D,B,C]])
        # B D E
        t2 = np.array([[B,D,E]])
        
        faces = np.concatenate([faces, t1, t2])
        faces[sampled_face] = [A,D,C]
        faces[second_face] = [A,E,D]
        
        if face_tracker is not None:
            current_tracker = np.arange(len(faces))
            current_tracker[sampled_face] = sampled_face
            current_tracker[second_face] = second_face
            current_tracker[-2] = sampled_face
            current_tracker[-1] = second_face
            
            face_tracker.update_tracker(current_tracker)
    else:
        # D B C
        t1 = np.array([[D,B,C]])
        
        faces = np.concatenate([faces, t1])
        faces[sampled_face] = [A,D,C]
        
        if face_tracker is not None:
            current_tracker = np.arange(len(faces))
            current_tracker[sampled_face] = sampled_face
            current_tracker[-1] = sampled_face
            
            face_tracker.update_tracker(current_tracker)

    return vertices, faces, D

def add_single_point(vertices, faces, sampled_point, sampled_face, face_tracker = None):
    # adds a single point to the mesh
    # if collinear, it adds it by splitting the corresponding edge
    
    # WARNING: you should check if any point is equal to a vertex!!
    
    A = faces[sampled_face, 0]
    B = faces[sampled_face, 1]
    C = faces[sampled_face, 2]

    # check if point is collinear to some edge
    if np.any([collinear(vertices[A], vertices[B], sampled_point), collinear(vertices[B], vertices[C], sampled_point), collinear(vertices[C], vertices[A], sampled_point)]):
        which_coll=np.argmin(np.array([collinear(vertices[A], vertices[B], sampled_point, return_value=True), collinear(vertices[B], vertices[C], sampled_point, return_value=True), collinear(vertices[C], vertices[A], sampled_point, return_value=True)]))
        
        # adds the point to the closest edge
        if which_coll==0:
            return _add_collinear_point(A,B,C,vertices, faces, sampled_point, sampled_face, face_tracker=face_tracker)
        elif which_coll == 1:
            return _add_collinear_point(B,C,A,vertices, faces, sampled_point, sampled_face, face_tracker=face_tracker)
        elif which_coll==2:
            return _add_collinear_point(C,A,B,vertices, faces, sampled_point, sampled_face, face_tracker=face_tracker)
    
    # if here, the point is not collinear and can be added safely
    vertices, faces, added_pt = add_internal_points(vertices, faces, sampled_point[np.newaxis], sampled_face[np.newaxis], face_tracker=face_tracker)
    added_pt = added_pt[0]  # reduce dimensions
    return vertices, faces, added_pt

def _safe_add_points(vertices, faces, sampled_points, sampled_faces, face_tracker = None):
    # subroutine of add_points(), to safely add points (duh..)
    
    nfaces = len(faces)
    
    # check if several points need to be added to the same face
    _, unique_indices = np.unique(sampled_faces, return_index  = True)
    
    
    # save position of added points
    added_points = np.zeros(len(sampled_points), dtype = int)+len(vertices)
    added_points[unique_indices] += np.arange(len(unique_indices))
    
    # add only first occurence of points
    vertices, faces, _ = add_internal_points(vertices, faces, sampled_points[unique_indices], sampled_faces[unique_indices], face_tracker=face_tracker)
    
    # select points that need to be projected again
    left_out_points = np.setdiff1d(np.arange(len(sampled_points)), unique_indices)

    if len(left_out_points)>0:
        sampled_points = sampled_points[left_out_points]
        
        faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
        sampled_faces = closest_faces(sampled_points, vertices, faces[faces_to_check], return_faces=True)[1]
        sampled_faces = faces_to_check[sampled_faces]
        
        # sampled_facestrue = closest_faces(sampled_points, vertices, faces, return_faces=True)[1]
        
        
        vertices, faces, newly_added_pts = add_points(vertices, faces, sampled_points, sampled_faces, return_face_tracker=False, face_tracker = face_tracker)
        added_points[left_out_points] = newly_added_pts
    
    return vertices, faces, added_points

def compute_collinearity_matrix(vertices, faces, sampled_points, sampled_faces):
    # returns the collinearity matrix of every point
    # i.e. a (len(sampled_points), 3) boolean matrix with the following properties:
    #       - for each row, the number of True values is either one or zero
    #       - if collinearity_matrix[i, 0] is True, then sampled_points[i] is collinear with A[i] and B[i]
    #       - if collinearity_matrix[i, 1] is True, then sampled_points[i] is collinear with B[i] and C[i]
    #       - if collinearity_matrix[i, 2] is True, then sampled_points[i] is collinear with C[i] and A[i]
    
    A = faces[sampled_faces, 0]
    B = faces[sampled_faces, 1]
    C = faces[sampled_faces, 2]

    collinearity_matrix = np.stack([collinear(vertices[A], vertices[B], sampled_points), collinear(vertices[B], vertices[C], sampled_points), collinear(vertices[C], vertices[A], sampled_points)], axis = -1)
    return collinearity_matrix


def add_points(vertices, faces, sampled_points, sampled_faces, return_face_tracker = False, face_tracker = None):
    # adds points to the mesh "safely", i.e. by splitting edges and joining to closest vertex if necessary
    nfaces = len(faces)
    
    if return_face_tracker and (face_tracker is None):
        face_tracker = FaceTracker(nfaces)
    
    which_vertex = np.any(find_close_points(vertices, sampled_points), axis = -1)
    
    collinearity_matrix = compute_collinearity_matrix(vertices, faces, sampled_points, sampled_faces)
    which_collinear = np.any(collinearity_matrix, axis = -1)
    which_collinear[which_vertex] = False
    which_not_collinear = np.logical_not(which_collinear)
    which_not_collinear[which_vertex] = False
    
    
    # add normal_points
    vertices, faces, newly_added_points = _safe_add_points(vertices, faces, sampled_points[which_not_collinear], sampled_faces[which_not_collinear], face_tracker = face_tracker)
    
    
    # save position of added points
    added_points = np.zeros(len(sampled_points), dtype = int)
    added_points[which_not_collinear] = newly_added_points
    
    # old
    # which_collinear = np.sum(collinearity_matrix, axis = -1) == 1
    # which_vertex = np.sum(collinearity_matrix, axis = -1) == 2
    
    # take care of points very close to vertices
    if which_vertex.sum()>0:
        added_points[which_vertex] = np.argmin(np.linalg.norm(sampled_points[which_vertex][np.newaxis]-vertices[:,np.newaxis], axis = -1), axis = 0)
    
    if which_collinear.sum()>0:
        sampled_points = sampled_points[which_collinear]
        
        # sampled_facestrue = closest_faces(sampled_points, vertices, faces, return_faces=True)[1]
        faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
        
        sampled_faces = closest_faces(sampled_points, vertices, faces[faces_to_check], return_faces=True)[1]
        
        sampled_faces = faces_to_check[sampled_faces]
        
        # _, counts = np.unique(sampled_faces, return_counts  = True)
        # if np.any(counts>2):
        #     raise NotImplementedError('More than one collinear point in the same face, try remeshing or implement the logic to handle this.')
        
        
        which_collinear = np.nonzero(which_collinear)[0]
        
        nfaces = len(faces)
        
        # add sampled points and faces to the mesh
        for i in range(len(sampled_points)):
            # project on faces
            faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
            sampled_face = closest_faces(sampled_points[[i]], vertices, faces[faces_to_check], return_faces=True)[1][0]
            sampled_face = faces_to_check[sampled_face]
            
            # sampled_facetrue = closest_faces(sampled_points[[i]], vertices, faces, return_faces=True)[1][0]
            vertices, faces, added_pt = add_single_point(vertices, faces, sampled_points[i], sampled_face, face_tracker=face_tracker)
            added_points[which_collinear[i]] = added_pt
    
    if not return_face_tracker:
        return vertices, faces, added_points
    else:
        return vertices, faces, added_points, face_tracker


def closest_faces(points, vertices, faces, return_faces = False):
    # projects the input points on the mesh and returns the corresponding points and faces
    # if return_faces is True, the index of the face on which the point was projected is returned

    all_proj = project_pointcloud_on_faces(points, vertices, faces)
    picked_faces = np.linalg.norm(points[:,np.newaxis]-all_proj, axis = -1).argmin(axis = -1)
    
    # projected coordinates
    out = all_proj[np.arange(len(points)),picked_faces]
    
    if return_faces:
        out = (out, picked_faces)
    return out



################### vertex sampling #########################
def vertex_probability(vertices, faces):
    # face probabilities
    all_areas = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]])
    all_areas /= all_areas.sum()
    
    # vertex probability is the probability of all the faces it touches divided by 3
    all_prob = np.array(np.sum(trimesh.Trimesh(vertices=vertices, faces=faces).faces_sparse.multiply(np.broadcast_to(all_areas[np.newaxis], (len(vertices), len(faces)))), axis = -1))[:,0]
    all_prob /= all_prob.sum()  # all_prob.sum() is about 3
    
    return all_prob

def uniform_vertex_sampling(vertices, faces, num_points, remesh = False, return_indices = True, generator = None):
    """
    this uniform vertex sampling is approximate, each vertex is chosen with a probability proportional to the area of the faces it touches
    
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        num_points: int, number of points to sample
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    sampled_vertices = generator.choice(len(vertices), size = num_points, p = vertex_probability(vertices, faces), replace = False)

    if return_indices:
        return vertices, faces, sampled_vertices
    else:
        return vertices[sampled_vertices]
    
def sample_mesh_vertices(vertices, faces, npoints, faces_to_sample = None, generator = None, return_indices = True, replace = False):
    # sample npoints vertices uniformly on the input mesh
    # NOTE: this is done by sampling vertices based on neighbouring face area
    # if faces_to_sample is not None, points are only sampled in the subset of vertices adjacent to the specified faces
    # NOTE: if replace is False and the number of points to sample is greater than the sampling space, npoints is clipped to allow sampling without replacement



    if generator is None:
        generator = np.random.default_rng()

    if faces_to_sample is None:
        vertices_to_sample = np.arange(len(vertices))
    else:
        vertices_to_sample = np.unique(faces[faces_to_sample])
    
    
    all_probs = vertex_probability(vertices, faces)
    all_probs = all_probs[vertices_to_sample]
    all_probs /= all_probs.sum()
    
    if replace is False and  npoints > len(vertices_to_sample):
        npoints = len(vertices_to_sample)
    
    sampled_vertices = generator.choice(vertices_to_sample, size = npoints, p = all_probs, replace = replace)

    if return_indices:
        return vertices, faces, sampled_vertices
    else:
        return vertices[sampled_vertices]
    
def edge_distance_poisson_disk_vertex_sampling(vertices, faces, min_dist = None, num_points = None, points_to_sample = None, seed_vertices = None, remesh = False, generator = None, return_original_faces = False, verbose = False):
    """
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        min_dist: float, minimum distance between the points of the poisson sampling
        num_points: (optional) int, rough number of points to sample, if min_dist is None, this should be given
        points_to_sample: points to sample in each disk
        seed_vertices: (optional) list of int, index of the vertices that should be included in the final sampling
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
                NOTE: sometimes it does not work, try upsampling with something more robust
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    # get a rough estimate of the number of points needed to cover the mesh
    total_area = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).sum()
    if min_dist is not None:
        num_points = int(0.5*total_area/min_dist**2)  # rough estimate, the 0.5 is empirical
        if verbose:
            print(f'Sampling about {num_points} points with a minimum distance of {min_dist}')
            
    elif min_dist is None:
        assert num_points is not None, 'If min_dist is None, num_points must be not None'
        min_dist = np.sqrt(0.5*total_area/num_points)
        
        if verbose:
            print(f'Sampling about {num_points} points with an estimated minimum distance of {min_dist}')
    
    
    if points_to_sample is None:
        # heuristics
        points_to_sample = int(30*np.max([(triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).max()-np.pi*min_dist**2)/(3*np.pi*min_dist**2), 1]))
    
    if verbose:
        print(f'Number of points sampled in each disk: {points_to_sample}')
    
    
    # variable that determines whether the mesh is fine enough for an accurate sampling
    fine_mesh = edge_lengths(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).mean()/min_dist
    if verbose:
        print(f'Mesh is{'' if fine_mesh<1 else ' not'} fine enough, with a coefficient of {fine_mesh}')
    # mesh is fine if fine_mesh<<1 (i.e. if a sphere of radius min_dist around a typical vertex contains many faces)
    fine_mesh = fine_mesh<1
    
    Q = deque()

    if seed_vertices is None:
        # sample first point
        
        vertices, faces, sampled_dipoles = sample_mesh_vertices(vertices, faces, npoints = 1, faces_to_sample = None, generator = generator, return_indices = True, replace = False)
        sampled_dipoles = sampled_dipoles.tolist()
        
        Q.append(sampled_dipoles[0])
        
        if verbose:
            print(f'Starting sampling with vertex: {sampled_dipoles[0]}, with coordinates {vertices[sampled_dipoles[0]]}')
    else:
        if not isinstance(seed_vertices, np.ndarray):
            assert isinstance(seed_vertices, list), 'Seed vertices must be a list of mesh vertices'
        else:
            assert len(seed_vertices.shape) == 1 and seed_vertices.dtype == int, 'Seed vertices must be a list of mesh vertices'
        sampled_dipoles = []
        for v in seed_vertices:
            sampled_dipoles.append(v)
            Q.append(v)
        
        if verbose:
            print(f'Starting sampling with {len(seed_vertices)} seed vertices: {seed_vertices}, with coordinates {vertices[seed_vertices]}')


    iter = 0
    pbar = tqdm(total=num_points, desc="Poisson disk sampling points")
    while len(Q) > 0:
        pbar.set_description("Iteration %d" % (iter+1))
        # print(f'Iter: {iter}, dipole: {len(sampled_dipoles)}/{num_points}')
        
        # current point
        point = Q.popleft()
        
        # create circular submesh from which to sample
        full_circle_vertices, full_circle_faces, source, selected_faces, original_faces, original_vertices = compute_close_faces(vertices, faces, point, min_dist, fine_mesh = fine_mesh, method = 'any', return_original_vertices=True)
        
        # sample points
        full_circle_vertices, full_circle_faces, good_points = sample_mesh_vertices(full_circle_vertices, full_circle_faces, npoints = points_to_sample, faces_to_sample = selected_faces, generator = generator, return_indices = True, replace = False)
        
        # check if points are at the correct distance from current center
        geoalg = distance_graph(full_circle_vertices, full_circle_faces)
        dists = edge_distances(geoalg, [source], good_points)[0]
        good_points = good_points[(dists>min_dist)&(dists<2*min_dist)]
        
        if len(good_points)>0:
            # check which points are at the correct distance from each other
            dist_matrix = compute_graph_dist_matrix(geoalg, good_points)
            tokeep = [0]
            for i in range(1,len(good_points)):
                if dist_matrix[tokeep, i].min()>min_dist:
                    tokeep.append(i)
            good_points = good_points[tokeep]
        
        # convert good points to main mesh space
        good_points = original_vertices[good_points]
        
        ##### check if points are good wrt to main mesh

        # create submesh with radius 3*min_dist around current point to speed up computations
        # select a subset of vertices using euclidean distance
        dists = np.linalg.norm(vertices-vertices[point], axis = -1)
        selected_vertices = np.asarray(dists<3*min_dist).nonzero()[0]
        
        # create euclidean submesh
        selected_faces = faces[np.any(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)]
        selected_faces = select_connected_faces(selected_faces, point)
        submesh_vertices, submesh_faces, selected_vertices = create_submesh(vertices, selected_faces)
        
        # find already sampled points in submesh space
        _, points_indices, _ = np.intersect1d(selected_vertices, sampled_dipoles, assume_unique=True, return_indices=True)    
        old_points_to_check = np.arange(len(selected_vertices))[points_indices]
        
        # find points to test in submesh space (the difference with above is due to the fact that I already know that good_points is inside selected_vertices)
        points_indices = np.nonzero( selected_vertices == np.array(good_points)[:,np.newaxis] )[1]
        new_points_to_check = np.arange(len(selected_vertices), dtype = np.int32)[points_indices]
        
        # compute geodesic distances on the subset
        geoalg = distance_graph(submesh_vertices, submesh_faces)
        tokeep = []
        for i,p in enumerate(new_points_to_check):
            dists = edge_distances(geoalg, [p], old_points_to_check)[0]
            if dists.min()>min_dist:
                tokeep.append(good_points[i])
                Q.append(good_points[i])
        sampled_dipoles += tokeep
        
        ###################################
        
        # OLD
        # if len(tokeep)>=points_to_sample-max_rejections:
        #     raise ValueError('Too few rejections!!!')
        
        # update progress bar
        iter += 1
        pbar.update(len(tokeep))
    pbar.close()
    
    if return_original_faces:
        return vertices, faces, sampled_dipoles, original_sampled_faces
    else:
        return vertices, faces, sampled_dipoles


def poisson_disk_vertex_sampling(vertices, faces, min_dist = None, num_points = None, points_to_sample = None, seed_vertices = None, remesh = False, generator = None, return_indices = True, verbose = False):
    """
        vertices: array (n_vertices, 3), vertices array of the mesh
        faces: array (n_faces, 3), faces array of the mesh
        min_dist: float, minimum distance between the points of the poisson sampling
        num_points: (optional) int, rough number of points to sample, if min_dist is None, this should be given
        points_to_sample: points to sample in each disk
        seed_vertices: (optional) list of int, index of the vertices that should be included in the final sampling
        remesh: boolean, whether or not to apply a flat remeshing strategy. This will increase the quality of the sampling, but lower the speed of the algorithm
                NOTE: sometimes it does not work, try upsampling with something more robust
        generator: (optional) a numpy random generator object to control random sampling
    """
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    # get a rough estimate of the number of points needed to cover the mesh
    total_area = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).sum()
    if min_dist is not None:
        num_points = int(0.5*total_area/min_dist**2)  # rough estimate, the 0.5 is empirical
        if verbose:
            print(f'Sampling about {num_points} points with a minimum distance of {min_dist}')
            
    elif min_dist is None:
        assert num_points is not None, 'If min_dist is None, num_points must be not None'
        min_dist = np.sqrt(0.5*total_area/num_points)
        
        if verbose:
            print(f'Sampling about {num_points} points with an estimated minimum distance of {min_dist}')
    
    if points_to_sample is None:
        # heuristics
        points_to_sample = int(30*np.max([(triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).max()-np.pi*min_dist**2)/(3*np.pi*min_dist**2), 1]))
    
    if verbose:
        print(f'Number of points sampled in each disk: {points_to_sample}')
    
    
    # variable that determines whether the mesh is fine enough for an accurate sampling
    fine_mesh = edge_lengths(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).mean()/min_dist
    if verbose:
        print(f'Mesh is{'' if fine_mesh<1 else ' not'} fine enough, with a coefficient of {fine_mesh}')
    # mesh is fine if fine_mesh<<1 (i.e. if a sphere of radius min_dist around a typical vertex contains many faces)
    fine_mesh = fine_mesh<1
    
    Q = deque()

    if seed_vertices is None:
        # sample first point
        
        vertices, faces, sampled_dipoles = sample_mesh_vertices(vertices, faces, npoints = 1, faces_to_sample = None, generator = generator, return_indices = True, replace = False)
        sampled_dipoles = sampled_dipoles.tolist()
        
        Q.append(sampled_dipoles[0])
        
        if verbose:
            print(f'Starting sampling with vertex: {sampled_dipoles[0]}, with coordinates {vertices[sampled_dipoles[0]]}')
    else:
        if not isinstance(seed_vertices, np.ndarray):
            assert isinstance(seed_vertices, list), 'Seed vertices must be a list of mesh vertices'
        else:
            assert len(seed_vertices.shape) == 1 and seed_vertices.dtype == int, 'Seed vertices must be a list of mesh vertices'
        sampled_dipoles = []
        for v in seed_vertices:
            sampled_dipoles.append(v)
            Q.append(v)
        
        if verbose:
            print(f'Starting sampling with {len(seed_vertices)} seed vertices: {seed_vertices}, with coordinates {vertices[seed_vertices]}')


    iter = 0
    pbar = tqdm(total=num_points, desc="Poisson disk sampling points")
    while len(Q) > 0:
        pbar.set_description("Iteration %d" % (iter+1))
        # print(f'Iter: {iter}, dipole: {len(sampled_dipoles)}/{num_points}')
        
        # current point
        point = Q.popleft()
        
        # create circular submesh from which to sample
        full_circle_vertices, full_circle_faces, source, selected_faces, original_faces, original_vertices = compute_close_faces(vertices, faces, point, min_dist, fine_mesh = fine_mesh, method = 'any', return_original_vertices=True)
        
        # sample points
        full_circle_vertices, full_circle_faces, good_points = sample_mesh_vertices(full_circle_vertices, full_circle_faces, npoints = points_to_sample, faces_to_sample = selected_faces, generator = generator, return_indices = True, replace = False)
        
        # check if points are at the correct distance from current center
        geoalg = geodesic.PyGeodesicAlgorithmExact(full_circle_vertices, full_circle_faces)
        dists = geoalg.geodesicDistances([source], good_points)[0]
        good_points = good_points[(dists>min_dist)&(dists<2*min_dist)]

        if len(good_points)>0:
            # check which points are at the correct distance from each other
            dist_matrix = compute_dist_matrix(geoalg, good_points)
            tokeep = [0]
            for i in range(1,len(good_points)):
                if dist_matrix[tokeep, i].min()>min_dist:
                    tokeep.append(i)
            good_points = good_points[tokeep]
        
        # convert good points to main mesh space
        good_points = original_vertices[good_points]
        
        ##### check if points are good wrt to main mesh

        # create submesh with radius 3*min_dist around current point to speed up computations
        # select a subset of vertices using euclidean distance
        dists = np.linalg.norm(vertices-vertices[point], axis = -1)
        selected_vertices = np.asarray(dists<3*min_dist).nonzero()[0]
        
        # create euclidean submesh
        selected_faces = faces[np.any(np.any(selected_vertices == faces[...,np.newaxis], axis = -1), axis = -1)]
        selected_faces = select_connected_faces(selected_faces, point)
        submesh_vertices, submesh_faces, selected_vertices = create_submesh(vertices, selected_faces)
        
        # find already sampled points in submesh space
        _, points_indices, _ = np.intersect1d(selected_vertices, sampled_dipoles, assume_unique=True, return_indices=True)    
        old_points_to_check = np.arange(len(selected_vertices))[points_indices]
        
        # find points to test in submesh space (the difference with above is due to the fact that I already know that good_points is inside selected_vertices)
        points_indices = np.nonzero( selected_vertices == np.array(good_points)[:,np.newaxis] )[1]
        new_points_to_check = np.arange(len(selected_vertices), dtype = np.int32)[points_indices]
        
        # compute geodesic distances on the subset
        geoalg = geodesic.PyGeodesicAlgorithmExact(submesh_vertices, submesh_faces)
        tokeep = []
        for i,p in enumerate(new_points_to_check):
            dists = geoalg.geodesicDistances([p], old_points_to_check)[0]
            if dists.min()>min_dist:
                tokeep.append(good_points[i])
                Q.append(good_points[i])
        sampled_dipoles += tokeep
        
        ###################################
        
        # OLD
        # if len(tokeep)>=points_to_sample-max_rejections:
        #     raise ValueError('Too few rejections!!!')
        
        # update progress bar
        iter += 1
        pbar.update(len(tokeep))
    pbar.close()
    
    if return_indices:
        return vertices, faces, sampled_dipoles
    else:
        return vertices[sampled_dipoles]
