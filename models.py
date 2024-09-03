import numpy as np
from scipy.integrate import solve_ivp

# Class for dynamics over a graph cellular sheaf

class SheafDynamic:
    '''
    A class to implement and solve dynamics over graph cellular sheaves

    Parameters:
        GCS: SheafBuilder -> An instantiated Graph Cellular Sheaf
        alpha: float -> Parameter controlling diffusion 
        beta: float -> Parameter controlling diffusion
        x0: np.array -> A proper initializiation of private opinion
        T: int -> Time horizon considered for the trajectories
        timespan: int -> Number of timepoints considered for the trajectories
        U: list -> Set of stubborn agents
        Y: list -> Set of agents to be observed 

    Methods:
        --DYNAMICS METHODS--

        opinion_dynamic():
            Basic implementation of a sheaf laplacian diffusion with eventually restriction encoded in self.PU

        forcing_opinion_dynamic():
            Dynamic based on controller input and a set of observables

        expression_dynamic():
            "Learning to lie" dynamic

        --SOLVERS-- (They all call solve_ivp method from scipy.integrate)

        privateOpinionDynamicSolver():
            It solves the dynamic implemented by opinion_dynamic()

        forcingOpinionDynamicSolver():
            It solves the dynamic implemented by forcing_opinion_dynamic()

        expressionDynamicSolver():
            It solves the dynamic implemented by expression_dynamic()
    '''

    def __init__(
            self, 
            GCS, 
            alpha, 
            beta, 
            x0,
            T = 100,
            timespan = 100,
            U = [],
            Y = [],
        ):

        # Structure of the sheaf
        self.GCS = GCS

        # Parameter in the dynamics
        self.alpha = alpha
        self.beta = beta

        # Forcing and control agents
        self.U = U
        self.Y = Y

        # Useful projectors
        self.PU = np.ones_like(x0)
        for i in U:
            self.PU[i*self.GCS.d:(i+1)*self.GCS.d] = 0

        self.PB = (self.GCS.B != 0).astype('int32')

        # Define the forcing matrix B to be the identity supported on C0(U;F)
        self.B = np.zeros_like(self.GCS.L_f)
        for agent in self.U:
            self.B[agent*self.GCS.d:(agent+1)*self.GCS.d, agent*self.GCS.d:(agent+1)*self.GCS.d] = np.eye(self.GCS.d)

        # Define the observation matrix C to be the identity supported on C0(Y;F)
        self.C = np.zeros_like(self.GCS.L_f)
        for agent in self.Y:
            self.C[agent*self.GCS.d:(agent+1)*self.GCS.d, agent*self.GCS.d:(agent+1)*self.GCS.d] = np.eye(self.GCS.d)

        # Initial opinion
        self.x0 = x0

        # Considered timewindow
        self.T = T
        self.timespan = timespan 
        self.time_points = np.linspace(0,self. T, self.timespan)

    ##############################
    #### METHODS FOR DYNAMICS ####
    ##############################

    def opinion_dynamic(self, t, x):
        return -self.alpha * self.PU * (self.GCS.L_f @ x)

    def forcing_opinion_dynamic(self, t, state, u):
        size = self.GCS.L_f.shape[0]
        x = state[:size]
        dxdt = -self.alpha * self.GCS.L_f @ x + self.GCS.B @ u
        dydt = self.C @ x
        return np.concatenate([dxdt, dydt])

    def expression_dynamic(self, t, B_flatten):

        B = B_flatten.reshape(len(self.GCS.edges) * self.GCS.d, self.GCS.V * self.GCS.d)
        dt_dB = -self.beta * self.PB * (B @ np.outer(self.x0, self.x0))
        return dt_dB.flatten()
    
    ##############################
    #### SOLVERS FOR DYNAMICS ####
    ##############################
        
    def privateOpinionDynamicSolver(
            self,
        ):
        
        solution = solve_ivp(
            self.opinion_dynamic, 
            [0, self.T], 
            self.x0, 
            t_eval=self.time_points, 
            args=(),
            method='RK45'
            )
        
        trajectory = solution.y.T
        return trajectory

    def forcingOpinionDynamicSolver(
            self
        ):

        x0 = np.array(self.x0).flatten()
        y0 = np.copy(x0)
        u = np.random.randn(self.GCS.V*self.GCS.d)
        for agent in range(self.GCS.V):
            if agent not in self.U:
                u[agent*self.GCS.d:(agent+1)*self.GCS.d] = 0

        # Combine initial conditions for x and y into one state vector
        state_0 = np.concatenate([x0, y0])

        # Solve the combined ODE
        solution = solve_ivp(
            self.forcing_opinion_dynamic, 
            [0, self.T], 
            state_0, 
            t_eval=self.time_points, 
            args=(u,),
            method='RK45'
        )
        
        # Extract the trajectories for x and y
        trajectory_x = solution.y[:len(x0), :].T
        trajectory_y = solution.y[len(x0):, :].T

        return trajectory_x, trajectory_y

    def expressionDynamicSolver(
            self
        ):

        solution = solve_ivp(
            self.expression_dynamic, 
            [0, self.T], 
            self.GCS.B.flatten(), 
            t_eval=self.time_points, 
            args=(),
            method='RK45'
            )
        
        B_hat = solution.y[:,-1].reshape(len(self.GCS.edges)*self.GCS.d, self.GCS.V*self.GCS.d)
        
        # Tracker of the disagreement 
        Bs = solution.y.T.reshape(self.time_points.shape[0], len(self.GCS.edges)*self.GCS.d, self.GCS.V*self.GCS.d)
        disagreement = self.x0.T @ (Bs.transpose(0,2,1) @ Bs.transpose(0,1,2)) @ self.x0

        return B_hat, disagreement

# Class for dynamics over a simplicial cellular sheaf

class SimplicialSheafDynamic:
    '''
    A class to implement and solve dynamics over simplicial cellular sheaves

    Parameters:
        SC: SimplicialSheafBuilder -> An instantiated Simplicial Cellular Sheaf
        alpha: float -> Parameter controlling diffusion 
        beta: float -> Parameter controlling diffusion
        gamma: float -> Parameter controlling diffusion
        T: int -> Time horizon considered for the trajectories
        timespan: int -> Number of timepoints considered for the trajectories


    Methods:
        --DYNAMICS METHODS--

        edge_flow_dynamic():
            Basic implementation of a simplicial sheaf laplacian diffusion 

        expression_dynamic():
            Higher order learning to lie

        --SOLVERS-- (They all call solve_ivp method from scipy.integrate)

        edge_flow_solver():
            It solves the dynamic implemented by edge_flow_dynamic()

        expression_dynamic_solver():
            It solves the dynamic implemented by expression_dynamic()
    '''
    def __init__(
            self, 
            SC, 
            alpha, 
            beta,
            gamma,
            T = 100,
            timespan = 100,
        ):

        # Simplicial sheaf
        self.SC = SC
        self.PB_0 = (self.SC.B0 != 0).astype('int32')
        self.PB_1 = (self.SC.B1.T != 0).astype('int32')

        # Parameter in the dynamics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Considered timewindow
        self.T = T
        self.timespan = timespan 
        self.time_points = np.linspace(0,self. T, self.timespan)

    ##############################
    #### METHODS FOR DYNAMICS ####
    ##############################

    def edge_flow_dynamic(self, t, xi):
        return -self.alpha * self.SC.L1 @ xi   
     
    
    def expression_dynamic(self, t, B_flatten, xi0):

        # The two matrices are retrieved through proper indexing and reshaping of the concatenated array
        B0 = B_flatten[:self.SC.V*self.SC.d * len(self.SC.edges)*self.SC.d].reshape(self.SC.V * self.SC.d, len(self.SC.edges) * self.SC.d)
        B1 = B_flatten[self.SC.V*self.SC.d * len(self.SC.edges)*self.SC.d:].reshape(len(self.SC.triangles)*self.SC.d, len(self.SC.edges)*self.SC.d)
        
        # Expressions of the dynamics
        dtdB0 = (-self.beta * self.PB_0 * (B0 @ np.outer(xi0, xi0))).flatten()
        dtdB1 = (-self.gamma * self.PB_1 * (B1 @ np.outer(xi0, xi0))).flatten()

        return np.concatenate([dtdB0, dtdB1])
    
    ##############################
    #### SOLVERS FOR DYNAMICS ####
    ##############################

    def edge_flow_solver(
            self
        ):

        # Initial edge flow
        xi0 = self.SC.initial_edge_flow()

        solution = solve_ivp(
            self.edge_flow_dynamic, 
            [0, self.T], 
            xi0, 
            t_eval=self.time_points, 
            args=(),
            method='RK45'
            )

        return solution.y.T
    
    def expression_dynamic_solver(
            self
        ):

        xi0 = self.SC.initial_edge_flow()
        solution = solve_ivp(
            self.expression_dynamic, 
            [0, self.T], 
            np.concatenate([self.SC.B0.flatten(), self.SC.B1.T.flatten()]), 
            t_eval=self.time_points, 
            args=(xi0,),
            method='RK45'
            )
        
        # The trajectories for the lower and upper coboundary maps must be retrieved via proper indexing and reshaping
        B0_traj = solution.y[:self.SC.V*self.SC.d * len(self.SC.edges)*self.SC.d,:].T.reshape(self.time_points.shape[0], self.SC.V*self.SC.d, len(self.SC.edges)*self.SC.d)
        B1T_traj = solution.y[self.SC.V*self.SC.d * len(self.SC.edges)*self.SC.d:,:].T.reshape(self.time_points.shape[0], len(self.SC.triangles)*self.SC.d, len(self.SC.edges)*self.SC.d)

        return xi0, B0_traj, B1T_traj