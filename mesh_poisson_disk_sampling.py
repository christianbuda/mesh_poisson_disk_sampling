import numpy as np
from tqdm.notebook import tqdm
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


def add_points(vertices, faces, sampled_points, sampled_faces):
    # add sampled points and faces to the mesh
    
    # NOTE: this function can only add one point per triangle at a time!
    if np.any(np.unique(sampled_faces, return_counts = True)[1]>1):
        raise ValueError('add_points() only works when there is a single point per triangle to be added, found more than once here. Try refining the mesh or use _safe_add_points().')
    
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
    
    return vertices, faces


def sample_mesh_points(vertices, faces, npoints = 30, generator = None):
    # sample npoints uniformly on the input mesh and add vertices in those points
    # NOTE: this is done by sampling faces based on areas, and then by sampling uniformly inside the triangle
    #       it's only uniform if npoints << len(faces)
    # returns a new mesh in which the sampled vertices are the last npoints rows in vertices

    if generator is None:
        generator = np.random.default_rng()

    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]

    p = triangle_area(A, B, C)
    p /= p.sum()
    sampled_faces = generator.choice(faces.shape[0], size = npoints, p = p, replace=False)

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

    return add_points(vertices, faces, sampled_points, sampled_faces)

def points_from_coeffs(vertices, faces, sampled_faces, sampled_coeff):
    # computes sampled points coordinates from trilinear local coordinates values
    return np.sum(vertices[faces[sampled_faces]]*np.broadcast_to(sampled_coeff[...,np.newaxis], (len(sampled_faces),3,3)), axis = 1)


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

def select_connected_faces(faces, source):
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
            return(faces[list(g)])
    
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

def sample_submesh_points(vertices, faces, faces_to_sample, npoints = 30, generator = None):
    # sample npoints on the input mesh and add vertices in those points
    # returns a new mesh in which the sampled vertices are the last npoints rows in vertices

    if generator is None:
        generator = np.random.default_rng()

    A = vertices[faces[faces_to_sample,0]]
    B = vertices[faces[faces_to_sample,1]]
    C = vertices[faces[faces_to_sample,2]]

    p = triangle_area(A, B, C)
    p /= p.sum()
    sampled_faces = generator.choice(faces_to_sample, size = npoints, p = p, replace = False)

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
    

    return add_points(vertices, faces, sampled_points, sampled_faces), (sampled_faces, sampled_coeff)


def compute_dist_matrix(geoalg, vertices):
    # compute distance matrix between list of vertices
    # given PyGeodesicAlgorithmExact object in input
    
    nverts = len(vertices)
    out = np.zeros((nverts, nverts))
    
    # fill upper triangular part of matrix
    for idx in range(nverts-1):
        out[idx, idx+1:] = geoalg.geodesicDistances([vertices[idx]], vertices[idx+1:])[0]
    
    # simmetrize distance matrix
    i_lower = np.tril_indices(nverts, -1)
    out[i_lower] = out.T[i_lower]
    
    return out

def poisson_disk_sampling(vertices, faces, min_dist, num_dip = None, seed_vertices = None, remesh = True, generator = None):
    # min_dist: minimum distance in mm, if None it is estimated from num_dip
    
    
    if generator is None:
        generator = np.random.default_rng()
        
    if remesh == True:
        print('Performing flat remeshing on input mesh...')
        vertices, faces = flat_remeshing(vertices, faces)

    # get a rough estimate of the number of points needed to cover the mesh
    total_area = triangle_area(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]).sum()
    if min_dist is not None:
        num_dip = int(0.5*total_area/min_dist**2)  # rough estimate, the 0.5 is empirical
    elif min_dist is None:
        assert num_dip is not None, 'If min_dist is None, num_dip must be not None'
        min_dist = np.sqrt(0.5*total_area/num_dip)
        
        
    Q = deque()

    if seed_vertices is None:
        # sample first point
        vertices, faces = sample_mesh_points(vertices, faces, npoints = 1, generator = generator)

        sampled_dipoles = [vertices.shape[0]-1]

        Q.append(vertices.shape[0]-1)
    else:
        assert isinstance(seed_vertices, list), 'Seed vertices must be a list of mesh vertices'
        for v in seed_vertices:
            sampled_dipoles.append(v)
            Q.append(v)


    iter = 0
    pbar = tqdm(total=num_dip, desc="Poisson disk sampling points")
    while len(Q) > 0:
        pbar.set_description("Iteration %d" % (iter+1))
        # print(f'Iter: {iter}, dipole: {len(sampled_dipoles)}/{num_dip}')
        
        # current point
        point = Q.popleft()
        
        # create circular submesh from which to sample
        full_circle_vertices, full_circle_faces, source, selected_faces, original_faces = get_submesh_faces(vertices, faces, point, min_dist, method = 'any')
        points_to_sample = len(selected_faces)
        
        # sample points
        (full_circle_vertices, full_circle_faces), (sampled_faces, sampled_coeff) = sample_submesh_points(full_circle_vertices, full_circle_faces, selected_faces, npoints = points_to_sample, generator = generator)
        
        # convert sampled_faces to main mesh space
        sampled_faces = original_faces[sampled_faces]
        
        # check if points are at the correct distance from current center
        geoalg = geodesic.PyGeodesicAlgorithmExact(full_circle_vertices, full_circle_faces)
        good_points = full_circle_vertices.shape[0] - np.arange(points_to_sample, 0, -1)
        dists = geoalg.geodesicDistances([source], good_points)[0]
        good_points = good_points[(dists>min_dist)&(dists<2*min_dist)]
        
        # check which points are at the correct distance from each other
        dist_matrix = compute_dist_matrix(geoalg, good_points)
        tokeep = [0]
        for i in range(1,len(good_points)):
            if dist_matrix[tokeep, i].min()>min_dist:
                tokeep.append(i)
        good_points = good_points[tokeep]-full_circle_vertices.shape[0]+points_to_sample
        
        # add the good points so far to the main mesh
        sampled_points = points_from_coeffs(vertices, faces, sampled_faces[good_points], sampled_coeff[good_points])
        vertices, faces = add_points(vertices, faces, sampled_points, sampled_faces[good_points])
        good_points = np.arange(vertices.shape[0]-len(good_points), vertices.shape[0]).tolist()
        
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
    
    
    return vertices, faces, sampled_dipoles