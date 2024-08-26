import numpy as np
from scipy.linalg import null_space

class SheafBuilder:
    
    def __init__(
            self, 
            V, 
            d,
            cutoff = 0.5,
            theta = 0.9,
            constant = True,
            stubborn = None,
            control = None
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

        # Graph builder
        self.randomGraph()

        # Sheaf builder
        self.GCSbuilder()

    def randomGraph(
            self
        ):

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