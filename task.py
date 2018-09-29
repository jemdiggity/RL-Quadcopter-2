import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 22
        self.action_low = 325 #400-500 flies, 400 doesn't
        self.action_high = 425
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, rotor_speeds):
#         """Uses current pose of sim to return reward."""
        pose_error = abs(self.target_pos - self.sim.pose[:3])

        distance = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        avg_angle = np.mean(self.sim.pose[3:])
        linear_accel = np.linalg.norm(self.sim.linear_accel)
        angular_accels = np.linalg.norm(self.sim.angular_accels)

        reward = 0
#         reward -= 10 * pose_error[2]
#         reward -= 10 * pose_error[1]
#         reward -= 10 * pose_error[0]
#         reward -= 0.1 * linear_accel.sum()
#         reward -= 0.1 * angular_accels.sum()
        # reward -= 1.0 * np.std(rotor_speeds)
        # reward -= 1.0 * avg_angle
#         print('rotor speed {}'.format(np.std(rotor_speeds)))

        reward += 1000
        reward -= 0.05 * (distance ** 2)

        return reward / 1000.0

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        # make sure it's an actual array
        rotor_speeds = np.array(rotor_speeds)
        #rescale from -1 to +1 to full range
        rotor_speeds = (self.action_high - self.action_low)/2 * (rotor_speeds + 1) + self.action_low
        rotor_speeds = np.ones_like(rotor_speeds) * rotor_speeds[0]
        # print('\r rotor speeds {}'.format(rotor_speeds), end='')
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
            pose_all.append(np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v, self.sim.linear_accel, self.sim.angular_accels, self.sim.prop_wind_speed]))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v, self.sim.linear_accel, self.sim.angular_accels, self.sim.prop_wind_speed] * self.action_repeat)
        return state

    def dump(self):
        print("time {:4.1f} x {:=+04.1f} y {:=+04.1f} z {:=+04.1f}".format(self.sim.time, self.sim.pose[0], self.sim.pose[1], self.sim.pose[2]))
