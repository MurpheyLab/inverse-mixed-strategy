"""
Generate synthetic dataset using iLQGames (5 agents)
Author: Max Muchen Sun
"""

import jax 
jax.config.update("jax_default_device", jax.devices(backend='cpu')[0])
from jax import jit, vmap
import jax.numpy as jnp 

from dynax import DynSystem

import numpy as np
from functools import partial
import time
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os
from tqdm import tqdm 

agent_radius = 0.5
obst_radius = 0.5
scene_radius = 4.0
v_min = 0.3
v_max = 1.2
num_agents = 4

class GameAgent(DynSystem):
    def __init__(self,
                    dt: float, 
                    tsteps: int, 
                    x_dim: int, 
                    u_dim: int, 
                    Q: jnp.array, 
                    R: jnp.array) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q, R)
        self.risk_w = 1.0
    
    def set_goal(self, goal: jnp.array) -> None:
        self.terminal_extra = goal

    @partial(jit, static_argnums=(0,))
    def dyn(self, xt, ut) -> jnp.array:
        px, py, th, v = xt 
        w, a = ut
        xdot = jnp.array([
            v * jnp.cos(th),
            v * jnp.sin(th),
            w,
            a
        ])
        return xdot 
    
    def risk(self, x, extra):
        extra1, extra2, extra3, extra4, w, sd, goal, risk_w = extra
        
        dist1 = jnp.sqrt(jnp.sum(jnp.square(x[:2]-extra1[:2])))
        val1 = jnp.square(jnp.minimum(dist1 - 2.0 * agent_radius - (sd+0.2), 0.0)) 

        dist2 = jnp.sqrt(jnp.sum(jnp.square(x[:2]-extra2[:2])))
        val2 = jnp.square(jnp.minimum(dist2 - 2.0 * agent_radius - (sd+0.2), 0.0)) 

        dist3 = jnp.sqrt(jnp.sum(jnp.square(x[:2]-extra3[:2])))
        val3 = jnp.square(jnp.minimum(dist3 - 2.0 * agent_radius - (sd+0.2), 0.0)) 

        dist4 = jnp.sqrt(jnp.sum(jnp.square(x[:2]-extra4[:2])))
        val4 = jnp.square(jnp.minimum(dist4 - 2.0 * agent_radius - (sd+0.2), 0.0)) 

        return (val1 + val2 + val3 + val4) * w
    
    def vel_constraint(self, x):
        val1 = jnp.square(jnp.minimum(x[3]-v_min, 0.0))
        val2 = jnp.square(jnp.minimum(v_max-x[3], 0.0))
        return (val1 + val2) * 10000.0

    def runtime_loss_step(self, xt, ut, extra):
        extra1, extra2, extra3, extra4, w, sd, goal, risk_w = extra

        safety_loss = self.risk(xt[:2], extra) * risk_w
        ctrl_loss = jnp.sum(jnp.square(ut) * jnp.array([2.0, 1.0]))
        vel_loss = self.vel_constraint(xt)
        nav_loss = jnp.sum(jnp.square(xt-goal) * jnp.array([10.0, 10.0, 10.1, 10.01]))
        return safety_loss + ctrl_loss + vel_loss + nav_loss
    
    @partial(jit, static_argnums=(0,))
    def runtime_loss(self, x_traj, u_traj, extra_traj):
        acccum_loss = self.dt * jnp.sum(vmap(self.runtime_loss_step, in_axes=(0,0,0))(x_traj, u_traj, extra_traj))
        return acccum_loss # + traj_len
    
    @partial(jit, static_argnums=(0,))
    def terminal_loss(self, xt, goal):
        return jnp.sum(jnp.square(xt-goal) * jnp.array([100.0, 100.0, 100.0, 100.0]))


dt = 0.1
tsteps = 40
x_dim = 4 
u_dim = 2
Q = jnp.diag(jnp.array([2.0, 2.0, 0.1, 0.1]))
R = jnp.diag(jnp.array([2.0, 1.0]))

agent_1 = GameAgent(
    dt=dt, tsteps=tsteps, 
    x_dim=x_dim, u_dim=u_dim,
    Q=Q, R=R
)
agent_2 = GameAgent(
    dt=dt, tsteps=tsteps, 
    x_dim=x_dim, u_dim=u_dim,
    Q=Q, R=R
)
agent_3 = GameAgent(
    dt=dt, tsteps=tsteps, 
    x_dim=x_dim, u_dim=u_dim,
    Q=Q, R=R
)
agent_4 = GameAgent(
    dt=dt, tsteps=tsteps, 
    x_dim=x_dim, u_dim=u_dim,
    Q=Q, R=R
)
agent_5 = GameAgent(
    dt=dt, tsteps=tsteps, 
    x_dim=x_dim, u_dim=u_dim,
    Q=Q, R=R
)


@jit 
def pt2line(line_pt1, line_pt2, pt):    
    line_vec = line_pt2 - line_pt1
    pt_vec = pt - line_pt1
    scalar_proj = jnp.dot(pt_vec, line_vec) / jnp.dot(line_vec, line_vec)
    proj_pt = line_pt1 + scalar_proj * line_vec
    
    return proj_pt

@jit
def get_ref_traj(curr_state, init_state, goal_state):
    proj_pt = curr_state[:2]
    ref_vel = goal_state[:2] - proj_pt
    ref_vel = ref_vel / jnp.linalg.norm(ref_vel) * init_state[3]
    ref_angle = jnp.arctan2(ref_vel[1], ref_vel[0])
    ref_traj = proj_pt + jnp.arange(1,tsteps+1)[:,None] * dt * ref_vel
    ref_traj = jnp.concatenate([ref_traj, jnp.ones((tsteps,1)) * jnp.array([ref_angle, init_state[3]])], axis=1)
    return ref_traj


def test(idx):
    def wrap_to_pi(angle):
        return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    seed = idx
    # print(f'\nseed: {seed}')
    key = jax.random.PRNGKey(seed)

    vel = jax.random.uniform(
        key, minval=0.9, maxval=1.1
    )
    
    def sample_x_init(key):
        angle = jax.random.uniform(
            key, minval=-jnp.pi, maxval=jnp.pi
        )   
        
        return jnp.array([
            jnp.cos(angle) * scene_radius, 
            jnp.sin(angle) * scene_radius,
            wrap_to_pi(-jnp.pi + angle),
            vel
        ])
    
    dist_threshold = agent_radius * 3.5

    key, subkey = jax.random.split(key)
    x_init_1 = sample_x_init(key)

    while True:
        key, subkey = jax.random.split(key)
        x_init_2 = sample_x_init(key)
        dist = jnp.linalg.norm(x_init_1[:2] - x_init_2[:2])
        if dist > dist_threshold:
            break

    while True:
        key, subkey = jax.random.split(key)
        x_init_3 = sample_x_init(key)

        dist1 = jnp.linalg.norm(x_init_1[:2] - x_init_3[:2])
        dist2 = jnp.linalg.norm(x_init_2[:2] - x_init_3[:2])
        dist = jnp.min(jnp.array([dist1, dist2]))
        
        if dist > dist_threshold:
            break

    while True:
        key, subkey = jax.random.split(key)
        x_init_4 = sample_x_init(key)

        dist1 = jnp.linalg.norm(x_init_1[:2] - x_init_4[:2])
        dist2 = jnp.linalg.norm(x_init_2[:2] - x_init_4[:2])
        dist3 = jnp.linalg.norm(x_init_3[:2] - x_init_4[:2])
        dist = jnp.min(jnp.array([dist1, dist2, dist3]))
        
        if dist > dist_threshold:
            break

    while True:
        key, subkey = jax.random.split(key)
        x_init_5 = sample_x_init(key)

        dist1 = jnp.linalg.norm(x_init_1[:2] - x_init_5[:2])
        dist2 = jnp.linalg.norm(x_init_2[:2] - x_init_5[:2])
        dist3 = jnp.linalg.norm(x_init_3[:2] - x_init_5[:2])
        dist4 = jnp.linalg.norm(x_init_4[:2] - x_init_5[:2])
        dist = jnp.min(jnp.array([dist1, dist2, dist3, dist4]))
        
        if dist > dist_threshold:
            break
        
    min_w = 1000.0
    max_w = 1500.0
    key, subkey = jax.random.split(key)
    w1 = jax.random.uniform(key, minval=min_w, maxval=max_w)
    w1_traj = jnp.ones(tsteps)[:,None] * w1

    key, subkey = jax.random.split(key)
    w2 = jax.random.uniform(key, minval=min_w, maxval=max_w)
    w2_traj = jnp.ones(tsteps)[:,None] * w2
    
    key, subkey = jax.random.split(key)
    w3 = jax.random.uniform(key, minval=min_w, maxval=max_w)
    w3_traj = jnp.ones(tsteps)[:,None] * w3
    
    key, subkey = jax.random.split(key)
    w4 = jax.random.uniform(key, minval=min_w, maxval=max_w)
    w4_traj = jnp.ones(tsteps)[:,None] * w4

    key, subkey = jax.random.split(key)
    w5 = jax.random.uniform(key, minval=min_w, maxval=max_w)
    w5_traj = jnp.ones(tsteps)[:,None] * w5

    key, subkey = jax.random.split(key)
    sd1 = 0.5
    sd1_traj = jnp.ones(tsteps)[:,None] * sd1
    key, subkey = jax.random.split(key)
    sd2 = 0.5
    sd2_traj = jnp.ones(tsteps)[:,None] * sd2
    sd3 = 0.5
    sd3_traj = jnp.ones(tsteps)[:,None] * sd3
    sd4 = 0.5
    sd4_traj = jnp.ones(tsteps)[:,None] * sd4
    sd5 = 0.5
    sd5_traj = jnp.ones(tsteps)[:,None] * sd5
    
    goal_1 = jnp.array([
        -x_init_1[0], -x_init_1[1], x_init_1[2], x_init_1[3]
    ])
    goal_2 = jnp.array([
        -x_init_2[0], -x_init_2[1], x_init_2[2], x_init_2[3]
    ])
    goal_3 = jnp.array([
        -x_init_3[0], -x_init_3[1], x_init_3[2], x_init_3[3]
    ])
    goal_4 = jnp.array([
        -x_init_4[0], -x_init_4[1], x_init_4[2], x_init_4[3]
    ])
    goal_5 = jnp.array([
        -x_init_5[0], -x_init_5[1], x_init_5[2], x_init_5[3]
    ])

    hist_traj_1 = []
    hist_traj_2 = []
    hist_traj_3 = []
    hist_traj_4 = []
    hist_traj_5 = []

    goal_threshold = agent_radius
    
    u_traj_1 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
    u_traj_2 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
    u_traj_3 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
    u_traj_4 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
    u_traj_5 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))

    w_traj_one = jnp.ones((tsteps, 1))
    w_traj_zero = jnp.zeros((tsteps, 1))

    x_state_1 = x_init_1.copy()
    x_state_2 = x_init_2.copy()
    x_state_3 = x_init_3.copy()
    x_state_4 = x_init_4.copy()
    x_state_5 = x_init_5.copy()

    for t in range (200):
        dist2goal_1 = jnp.linalg.norm(goal_1[:2] - x_state_1[:2])
        dist2goal_2 = jnp.linalg.norm(goal_2[:2] - x_state_2[:2])
        dist2goal_3 = jnp.linalg.norm(goal_3[:2] - x_state_3[:2])
        dist2goal_4 = jnp.linalg.norm(goal_4[:2] - x_state_4[:2])
        dist2goal_5 = jnp.linalg.norm(goal_5[:2] - x_state_5[:2])

        if np.max([dist2goal_1, dist2goal_2, dist2goal_3, dist2goal_4, dist2goal_5]) < goal_threshold:
            break

        ref_traj_1 = get_ref_traj(x_state_1, x_init_1, goal_1)
        ref_traj_2 = get_ref_traj(x_state_2, x_init_2, goal_2)
        ref_traj_3 = get_ref_traj(x_state_3, x_init_3, goal_3)
        ref_traj_4 = get_ref_traj(x_state_4, x_init_4, goal_4)
        ref_traj_5 = get_ref_traj(x_state_5, x_init_5, goal_5)

        u_traj_1 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
        u_traj_2 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
        u_traj_3 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
        u_traj_4 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))
        u_traj_5 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps,1))

        step_list = jnp.power(10.0, jnp.linspace(-4.0, 0.0, 100))
        for iter in range(40):
            x_traj_1 = agent_1.dyn_scan(x_state_1, u_traj_1)
            x_traj_2 = agent_2.dyn_scan(x_state_2, u_traj_2)
            x_traj_3 = agent_3.dyn_scan(x_state_3, u_traj_3)
            x_traj_4 = agent_4.dyn_scan(x_state_4, u_traj_4)
            x_traj_5 = agent_5.dyn_scan(x_state_5, u_traj_5)

            v_traj_1 = agent_1.lqr_descent(x_state_1, u_traj_1, (x_traj_2, x_traj_3, x_traj_4, x_traj_5, w1_traj, sd1_traj, ref_traj_1, w_traj_one), ref_traj_1[-1])
            v_traj_2 = agent_2.lqr_descent(x_state_2, u_traj_2, (x_traj_1, x_traj_3, x_traj_4, x_traj_5, w2_traj, sd2_traj, ref_traj_2, w_traj_one), ref_traj_2[-1])
            v_traj_3 = agent_3.lqr_descent(x_state_3, u_traj_3, (x_traj_1, x_traj_2, x_traj_4, x_traj_5, w3_traj, sd3_traj, ref_traj_3, w_traj_one), ref_traj_3[-1])
            v_traj_4 = agent_4.lqr_descent(x_state_4, u_traj_4, (x_traj_1, x_traj_2, x_traj_3, x_traj_5, w4_traj, sd4_traj, ref_traj_4, w_traj_one), ref_traj_4[-1])
            v_traj_5 = agent_5.lqr_descent(x_state_5, u_traj_5, (x_traj_1, x_traj_2, x_traj_3, x_traj_4, w5_traj, sd5_traj, ref_traj_5, w_traj_one), ref_traj_5[-1])

            u_traj_1, opt_step_1 = agent_1.naive_line_search(step_list, x_state_1, u_traj_1, v_traj_1, (x_traj_2, x_traj_3, x_traj_4, x_traj_5, w1_traj, sd1_traj, ref_traj_1, w_traj_one), ref_traj_1[-1])
            u_traj_2, opt_step_2 = agent_2.naive_line_search(step_list, x_state_2, u_traj_2, v_traj_2, (x_traj_1, x_traj_3, x_traj_4, x_traj_5, w2_traj, sd2_traj, ref_traj_2, w_traj_one), ref_traj_2[-1])
            u_traj_3, opt_step_3 = agent_3.naive_line_search(step_list, x_state_3, u_traj_3, v_traj_3, (x_traj_1, x_traj_2, x_traj_4, x_traj_5, w3_traj, sd3_traj, ref_traj_3, w_traj_one), ref_traj_3[-1])
            u_traj_4, opt_step_4 = agent_4.naive_line_search(step_list, x_state_4, u_traj_4, v_traj_4, (x_traj_1, x_traj_2, x_traj_3, x_traj_5, w4_traj, sd4_traj, ref_traj_4, w_traj_one), ref_traj_4[-1])
            u_traj_5, opt_step_5 = agent_5.naive_line_search(step_list, x_state_5, u_traj_5, v_traj_5, (x_traj_1, x_traj_2, x_traj_3, x_traj_4, w5_traj, sd5_traj, ref_traj_5, w_traj_one), ref_traj_5[-1])

        hist_traj_1.append(x_state_1.copy())
        hist_traj_2.append(x_state_2.copy())
        hist_traj_3.append(x_state_3.copy())
        hist_traj_4.append(x_state_4.copy())
        hist_traj_5.append(x_state_5.copy())        

        if dist2goal_1 > goal_threshold:
            x_state_1, _ = agent_1.dyn_step(x_state_1, u_traj_1[0])
        if dist2goal_2 > goal_threshold:
            x_state_2, _ = agent_2.dyn_step(x_state_2, u_traj_2[0])
        if dist2goal_3 > goal_threshold:
            x_state_3, _ = agent_3.dyn_step(x_state_3, u_traj_3[0])
        if dist2goal_4 > goal_threshold:
            x_state_4, _ = agent_4.dyn_step(x_state_4, u_traj_4[0])
        if dist2goal_5 > goal_threshold:
            x_state_5, _ = agent_5.dyn_step(x_state_5, u_traj_5[0])
    
    hist_traj_1 = jnp.array(hist_traj_1)
    hist_traj_2 = jnp.array(hist_traj_2)
    hist_traj_3 = jnp.array(hist_traj_3)
    hist_traj_4 = jnp.array(hist_traj_4)
    hist_traj_5 = jnp.array(hist_traj_5)

    final_x_traj_combo = np.array([
        hist_traj_1, hist_traj_2, hist_traj_3, hist_traj_4, hist_traj_5
    ])
    x_init_combo = np.array([
        x_init_1, x_init_2, x_init_3, x_init_4, x_init_5
    ])
    goal_combo = np.array([
        goal_1, goal_2, goal_3, goal_4, goal_5
    ])
    params = np.array([
        [w1, sd1], 
        [w2, sd2],
        [w3, sd3],
        [w4, sd4],
        [w5, sd5]
    ])

    trial_path = f'./synthetic_dastaset_5agents/trial_{idx:03d}/'
    os.makedirs(trial_path, exist_ok=True)
    np.save(trial_path + 'x_traj.npy', final_x_traj_combo)
    np.save(trial_path + 'x_init.npy', x_init_combo)
    np.save(trial_path + 'goal.npy', goal_combo)
    np.save(trial_path + 'params.npy', params)


num_trials = 100
for idx in tqdm(range(0, num_trials)):
    test(idx)
