import dmp_discrete_output_omega.DMPs_discrete as DMP_omega
import dmp_discrete_ouput.DMPs_discrete as DMP_out


path = np.loadtxt(args.trajectory).T
dmp = DMPs_discrete(n_dmps=5,n_bfs=10000, dt=1/200)
dmp.imitate_path(y_des=path)
y_track, dy_track, ddy_track = dmp.rollout()
omega = dmp.w

