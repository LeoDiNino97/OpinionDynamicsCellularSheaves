import numpy as np
from scipy.integrate import solve_ivp

class SheafDynamic:
    def __init__(
            self, 
            L_f, 
            B0,
            alpha, 
            beta, 
            E, 
            V, 
            d, 
            x0,
            T = 100,
            timespan = 100,
            U = [],
            Y = [],
        ):

        # Structure of the sheaf
        self.E = E
        self.V = V
        self.d = d

        # Laplacian and coboundary maps
        self.L_f = L_f
        self.B0 = B0

        # Parameter in the dynamics
        self.alpha = alpha
        self.beta = beta

        # Forcing and control agents
        self.U = U
        self.Y = Y

        # Useful projectors
        self.PU = np.ones_like(x0)
        for i in U:
            self.PU[i*self.d:(i+1)*self.d] = 0

        self.PB = (self.B0 != 0).astype('int32')

        # Define the forcing matrix B to be the identity supported on C0(U;F)
        self.B = np.zeros_like(self.L_f)
        for agent in self.U:
            self.B[agent*d:(agent+1)*d, agent*d:(agent+1)*d] = np.eye(self.d)

        # Define the observation matrix C to be the identity supported on C0(Y;F)
        self.C = np.zeros_like(self.L_f)
        for agent in Y:
            self.C[agent*d:(agent+1)*d, agent*d:(agent+1)*d] = np.eye(self.d)

        # Initial private opinion 
        self.x0 = x0

        # Considered timewindow
        self.T = T
        self.timespan = timespan 
        self.time_points = np.linspace(0,self. T, self.timespan)

    ##############################
    #### METHODS FOR DYNAMICS ####
    ##############################

    def opinion_dynamic(self, t, x):
        return -self.alpha * self.PU * (self.L_f @ x)

    def forcing_opinion_dynamic(self, t, state, u):
        size = self.L_f.shape[0]
        x = state[:size]
        dxdt = -self.alpha * self.L_f @ x + self.B @ u
        dydt = self.C @ x
        return np.concatenate([dxdt, dydt])

    def expression_dynamic(self, t, B_flatten):

        B = B_flatten.reshape(self.E * self.d, self.V * self.d)
        dt_dB = -self.beta * self.PB * (B @ np.outer(self.x0, self.x0))
        return dt_dB.flatten()
    
    ##############################
    #### SOLVERS FOR DYNAMICS ####
    ##############################
        
    def privateOpinionDynamicSolver(
            self
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
        u = np.random.randn(self.V*self.d)
        for agent in range(self.V):
            if agent not in self.U:
                u[agent*self.d:(agent+1)*self.d] = 0

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
            self.B0.flatten(), 
            t_eval=self.time_points, 
            args=(self.beta,),
            method='RK45'
            )
        
        B_hat = solution.y[:,-1].reshape(self.E*self.d, self.V*self.d)
        
        # Tracker of the disagreement 
        Bs = solution.y.T.reshape(self.timepoints.shape[0], self.E*self.d, self.V*self.d)
        disagreement = self.x0.T @ (Bs.transpose(0,2,1) @ Bs.transpose(0,1,2)) @ self.x0

        return B_hat, disagreement