import numpy as np
from scipy.linalg import null_space
from itertools import combinations

class SheafBuilder:
    
    def __init__(
            self, 
            V, 
            d,
            cutoff = 0.5,
            theta = 0.9,
            constant = True,
            stubborn = None,
            control = None,
            seed = 42
            ):

        self.V = V
        self.nodes = list(range(self.V))
        self.d = d
        self.cutoff = cutoff
        self.theta = theta

        self.constant = constant
        self.augmented = False

        # Subset of agents for forcing and control in dynamics
        self.stubborn = stubborn
        self.control = control

        # Seed for reproducibility
        self.seed = seed

        # Graph builder
        self.randomGraph()

        # Sheaf builder
        self.GCSbuilder()

    def randomGraph(
            self, 
        ):
        if self.seed is not None:
                np.random.seed(self.seed)

        self.edges = []

        points = np.random.rand(self.V, 2)

        self.A = np.zeros((self.V,self.V))

        for i in range(self.V):
            for j in range(i + 1, self.V):
                
                self.A[i,j] = np.linalg.norm(points[i,:] - points[j,:]) <= self.cutoff
                self.A[j,i] = np.linalg.norm(points[i,:] - points[j,:]) <= self.cutoff

                if self.A[i,j] == 1:
                    self.edges.append((i,j))

    def GCSbuilder(
            self
        ):

        E = len(self.edges)

        # Incidency linear maps
        if self.constant == True:
            self.F = {
                e:{
                    e[0]:np.eye(self.d),
                    e[1]:np.eye(self.d)
                    } 
                    for e in self.edges
                }               
        else:
            self.F = {
                e:{
                    e[0]:np.random.randn(self.d, self.d),
                    e[1]:np.random.randn(self.d, self.d)
                    } 
                    for e in self.edges
                }                  

        # Coboundary maps

        self.B = np.zeros((self.d*E, self.d*self.V))                        

        for i, edge in enumerate(self.edges):

            # Main loop to populate the coboundary map

            edge = self.edges[i]

            u = edge[0] 
            v = edge[1] 

            B_u = self.F[edge][u]
            B_v = self.F[edge][v]

            self.B[i*self.d:(i+1)*self.d, u*self.d:(u+1)*self.d] = B_u           
            self.B[i*self.d:(i+1)*self.d, v*self.d:(v+1)*self.d] = - B_v

        self.L_f = self.B.T @ self.B

    def initial_state(self): 
        if self.seed is not None:
            np.random.seed(self.seed)
            
        return np.random.randn(self.V*self.d)

    def forcing_opinion(self):
        
        u = np.random.randn(self.V*self.d)
        for agent in range(self.V):
            if agent not in self.stubborn:
                u[agent*self.d:(agent+1)*self.d] = 0
        
        return u
    
    
    def augmentedConstantSheaf(
            self,
            gamma = 0.1
        ):

        if not self.augmented:
            nodes = list(range(self.V))
            stubborn_parents = {node: self.V + node for node in nodes}

            self.edges += [(node, stubborn_parents[node]) for node in nodes]
            nodes += list(stubborn_parents.values())

            maps = {
                edge : {
                    edge[0]: np.eye(self.d),
                    edge[1]: np.eye(self.d)
                }
                for edge in self.edges
            }

            for node in nodes[:self.V]:
                maps[(node, stubborn_parents[node])][node] = gamma * np.eye(self.d)
                maps[(node, stubborn_parents[node])][stubborn_parents[node]] = gamma * np.eye(self.d)

            B = np.zeros((self.d*len(self.edges), self.d*len(nodes)))

            for i in range(len(self.edges)):
                edge = self.edges[i]

                u = edge[0] 
                v = edge[1] 

                B[i*self.d:(i+1)*self.d, u*self.d:(u+1)*self.d] = maps[edge][u]
                B[i*self.d:(i+1)*self.d, v*self.d:(v+1)*self.d] = - maps[edge][v]

            # Sheaf Laplacian
            L_f = B.T @ B

            # Attributes update
            self.nodes = nodes
            self.F = maps
            self.L_f = L_f
            self.B = B
            self.augmented = True

        else:
            print('The original sheaf has already been augmented!')

    def augmented_initial_state(
            self
    ):
        if not self.augmented: 
            return('The sheaf must be augmented before generating such an initial opinion distribution!')
        
        X0 = np.zeros(self.d * len(self.nodes))
        x0 = np.random.randn(self.d * int(len(self.nodes)/2))
        X0[int(len(self.nodes)/2)*self.d:] = x0
        X0[0:int(len(self.nodes)/2)*self.d] = x0
        
        return X0
    
    def null_space_projector(
            self, 
            x
    ):
        
        null = null_space(self.L_f)
        projector = null @ null.T
        
        return projector @ x
    

class SimplicialSheafBuilder:
    
    def __init__(
            self, 
            V, 
            d,
            ntriangles = 'Full',
            cutoff = 0.5,
            seed = 42
            ):

        self.V = V
        self.nodes = list(range(self.V))
        self.d = d
        self.ntriangles = ntriangles
        self.cutoff = cutoff

        # Seed for reproducibility
        self.seed = seed

        # Graph builder
        self.randomGraph()

        # Simplicial builder
        self.random2SC()

        # Sheaf builder
        self.SCSbuilder()

    def randomGraph(
            self
        ):

        if self.seed is not None:
            np.random.seed(self.seed)

        self.edges = []

        points = np.random.rand(self.V, 2)

        self.A = np.zeros((self.V,self.V))

        for i in range(self.V):
            for j in range(i + 1, self.V):
                
                self.A[i,j] = np.linalg.norm(points[i,:] - points[j,:]) <= self.cutoff
                self.A[j,i] = np.linalg.norm(points[i,:] - points[j,:]) <= self.cutoff

                if self.A[i,j] == 1:
                    self.edges.append((i,j))

    def triangleFinder(self):
        all_cliques = list(combinations(range(self.V), 3)) 
        true_cliques = []
        for clique in all_cliques:
            u, v, w = clique[0], clique[1], clique[2]
            A = ((u,v) in self.edges or (v,u) in self.edges)
            B = ((v,w) in self.edges or (w,v) in self.edges)
            C = ((w,u) in self.edges or (u,w) in self.edges)

            if A and B and C:
                true_cliques.append((u,v,w))

        return true_cliques
    
    def random2SC(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        all_triangles = self.triangleFinder()
        if self.ntriangles == 'Full':
            self.triangles = all_triangles
        else:
            T = [all_triangles[i] for i in np.random.choice(len(all_triangles), self.ntriangles, replace=False).tolist()]
            self.triangles = T
    
    def SCSbuilder(
            self
        ):

        # Coboundary maps

        self.B0 = np.zeros((self.d*self.V, self.d*len(self.edges)))               
        self.B0_graph = np.zeros((self.V, len(self.edges))) 

        self.B1 = np.zeros((self.d*len(self.edges), self.d*len(self.triangles)))
        
        edges_idxs = {}

        for i, edge in enumerate(self.edges):

            u = edge[0] 
            v = edge[1] 

            self.B0[u*self.d:(u+1)*self.d, i*self.d:(i+1)*self.d] = - np.eye(self.d)      
            self.B0[v*self.d:(v+1)*self.d, i*self.d:(i+1)*self.d] = np.eye(self.d)

            self.B0_graph[u, i] = -1
            self.B0_graph[v, i] = 1

            edges_idxs[edge] = i

        for j, triangle in enumerate(self.triangles):
                e1 = (triangle[0], triangle[1])
                e2 = (triangle[1], triangle[2])
                e3 = (triangle[0], triangle[2])

                self.B1[self.d*edges_idxs[e1]:self.d*(edges_idxs[e1]+1), j*self.d:(j+1)*self.d] = np.eye(self.d) 
                self.B1[self.d*edges_idxs[e2]:self.d*(edges_idxs[e2]+1), j*self.d:(j+1)*self.d] = - np.eye(self.d)
                self.B1[self.d*edges_idxs[e3]:self.d*(edges_idxs[e3]+1), j*self.d:(j+1)*self.d] = - np.eye(self.d)

        self.L0 = self.B0 @ self.B0.T
        self.L1 = self.B0.T @ self.B0 + self.B1 @ self.B1.T

    def initial_vertex_flow(self): 
        if self.seed is not None:
            np.random.seed(self.seed)
            
        return np.random.randn(self.V*self.d)
    
    def initial_edge_flow(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        return np.random.randn(len(self.edges)*self.d)