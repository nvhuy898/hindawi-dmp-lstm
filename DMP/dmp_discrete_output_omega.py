"""
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from DMP.dmp import DMPs

import numpy as np


class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    """ **kwargs allow you to pass KEYWORDED variable length of arguments(keyword arguments) to functions.
        You do not know how many arguments to the function. 
    
    """

    """
    n_dmps int: number of dynamic motor primitives        number of function y..=by*ay*(g-y)
    n_bfs int: number of basis functions per DMP         
    dt float: timestep for simulation 
    y0 list: initial state of DMPs 
    goal list: goal state of DMPs 
    w list: tunable parameters, control amplitude of basis functions 
    ay int: gain on attractor term y dynamics 
    by int: gain on attractor term y dynamics

    """

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern="discrete", **kwargs)

        self.gen_centers()
        # Set the centre of the Gaussian basis functions be spaced evenly throughout run time

        # set variance of Gaussian basis functions
        # trial and error to find this spacing

        # np.ones return a new array, filled with ones

        self.h = (np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax)  
        # self.n_bfs given
        # ** la phep luy thua                                                   # self.c
        # define in gen_centers(self)
        # phuong sai h                                                          # self.cs
        # define in class DMPs
        # part 1 page 6    part 1 page 9                                        # self.ax
        # define in class cs.py
        # self.cs
        # equal class cs.py

        self.check_offset()
        # Check to see if initial position and goal are the same if they are,
        # offset slightly so that the forcing term is not 0

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]
        """

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)  
        # np.linspace
        # return evenly spaced number
        # over a specified interval
        # tra ve cac so cach deu nhau trong 1 khoang nhat dinh

        self.c = np.ones(len(des_c))  # len() return the number of item

        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(
                -self.cs.ax * des_c[n])  
            # np.exp exponential of all elenments
            # part 1 page 16                                      # c[n] = exp(-ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        # print(x)
        return x * (self.goal[dmp_num] - self.y0[dmp_num])  #   x * (g - y)

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """
        # print(y_des)
        return np.copy(y_des[:, -1])  # np.copy return an array copy of the given object

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray): 
            x = x[:, None]  
            # isinstance(object, classinfo)
            # check object is a class, tuple os types
            # transpose matrix
        # print(x)
        # print(np.exp (-self.h * (x - self.c)**2))
        return np.exp(-self.h * (x - self.c) ** 2)  # ** la phep luy thua
        # psi in part 1 page 2
        # tinh mot gia tri x voi n_bfs gia tri c
    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # print(f_target)

        # calculate x and psi
        x_track = (self.cs.rollout())  # x_track --> 100                            
        # rollout() is an function in DMPs class
        # print(x_track)
        psi_track = self.gen_psi(x_track)  
        # print(psi_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))  # np.zeros return an array, filled
        for d in range(self.n_dmps):                  #          with zeros
            # spatial scaling term
            k = self.goal[d] - self.y0[d]
            for b in range(self.n_bfs):
                numer = np.sum( x_track * psi_track[:, b] * f_target[:, d])  
                # np.sum sum of array elenments over a
                # print(numer)
                denom = np.sum(x_track ** 2 * psi_track[:, b])  # given axis
                # print(denom)
                # x_track**2 --> ????
                self.w[d, b] = numer / (k * denom)  # tu va mau rut gon con 1 k
                # w --> ???   part 1 page 7
                # sT * s bang x_track**2
        self.w = np.nan_to_num(self.w)  
        # np.nan_to_num replace NAN with zero
        # nan Not a number and infinity with large finite numbers.                  
        # print(self.w)

    # test
    # def print_everything(self,**kwargs):
    #     print(self.y_des)
    #     print(self.y)
    #     print(self.dy)
    #     print(self.ddy)
    #     print(f_target)


# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--trajectory",type=str, help="toa do diem cuoi")
    parser.add_argument("-omg", "--omega",type=str, help="path file omega")
    parser.add_argument("-n", "--n_dmps",type=str, help="n_dmp")
    args=parser.parse_args()
    
    import matplotlib.pyplot as plt

    # load voi file .npz
    # path = np.load("Output_q_number1.npz")
    # load voi file .txt
#     print(args.trajectory)
    path = np.loadtxt(args.trajectory).T

    dmp = DMPs_discrete(n_dmps=int(args.n_dmps),n_bfs=10000, dt=1/200)
    dmp.imitate_path(y_des=path)
    y_track, dy_track, ddy_track = dmp.rollout()
    omega = dmp.w
    np.savetxt(args.omega,omega)
#     return omega

        

    


