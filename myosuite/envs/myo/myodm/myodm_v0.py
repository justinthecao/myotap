""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu), Vittorio Caggiano (caggiano@gmail.com)
Source  :: https://github.com/MyoHub/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import os
import time

import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.logger.reference_motion import ReferenceMotion
from myosuite.utils import gym
from myosuite.utils.quat_math import euler2quat, mat2quat, quat2euler, quatDiff2Vel
from myosuite.utils import gym
from myosuite.utils.quat_math import euler2quat, mat2quat, quat2euler, quatDiff2Vel

# ToDo
# - change target to reference


class TrackEnv(BaseV0):

    DEFAULT_CREDIT = """\
    MyoDex: A Generalizable Prior for Dexterous Manipulation
        Vittorio Caggiano, Sudeep Dasari, Vikash Kumar
        ICML-2023, https://arxiv.org/abs/2309.03130
    """

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 25.0,
        "penalty": 100,
    }

    def __init__(
        self, object_name, model_path, obsd_model_path=None, seed=None, **kwargs
    ):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.object_name = object_name
        time_stamp = str(time.time())

        # Process model_path to import the right object
        with open(curr_dir + model_path, "r") as file:
            processed_xml = file.read()
            processed_xml = processed_xml.replace("OBJECT_NAME", object_name)
        processed_model_path = (
            curr_dir + model_path[:-4] + time_stamp + "_processed.xml"
        )
        with open(processed_model_path, "w") as file:
            file.write(processed_xml)
        # Process obsd_model_path to import the right object
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            with open(curr_dir + obsd_model_path, "r") as file:
                processed_xml = file.read()
                processed_xml = processed_xml.replace("OBJECT_NAME", object_name)
            processed_obsd_model_path = (
                curr_dir + model_path[:-4] + time_stamp + "_processed.xml"
            )
            with open(processed_obsd_model_path, "w") as file:
                file.write(processed_xml)
        else:
            processed_obsd_model_path = None

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(
            self, object_name, model_path, obsd_model_path, seed, **kwargs
        )

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=processed_model_path,
            obsd_model_path=processed_obsd_model_path,
            seed=seed,
            env_credits=self.DEFAULT_CREDIT,
        )
        os.remove(processed_model_path)
        if (
            processed_obsd_model_path
            and processed_obsd_model_path != processed_model_path
        ):
            os.remove(processed_obsd_model_path)

        self.initialized_pos = False
        self._setup(**kwargs)

    def _setup(
        self,
        reference,  # reference target/motion for behaviors
        motion_start_time: float = 0,  # useful to skip initial motion
        motion_extrapolation: bool = True,  # Hold the last frame if motion is over
        obs_keys=DEFAULT_OBS_KEYS,
        weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
        Termimate_obj_fail=True,
        Termimate_pose_fail=False,
        **kwargs
    ):

        self.init_qpos = np.zeros(self.sim.model.nq)  # Initialize with proper size
    
     # Rest of your initialization code...
        self.ref = ReferenceMotion(
            reference_data=reference,
            motion_extrapolation=motion_extrapolation,
            random_generator=self.np_random,
        )
        self.motion_start_time = motion_start_time
        self.target_sid = self.sim.model.site_name2id("target")
        self.index_target_sid = self.sim.model.site_name2id("IFtip_target")
        ##########################################
        self.lift_bonus_thresh = 0.02
        ### PRE-GRASP
        self.obj_err_scale = 50
        self.base_err_scale = 40
        self.lift_bonus_mag = 1  # 2.5

        ### DEEPMIMIC
        self.qpos_reward_weight = 0.35
        self.qpos_err_scale = 5.0

        self.qvel_reward_weight = 0.05
        self.qvel_err_scale = 0.1

        # TERMINATIONS FOR OBJ TRACK
        self.obj_fail_thresh = 0.25
        # TERMINATIONS FOR HAND-OBJ DISTANCE
        self.base_fail_thresh = 0.25
        self.TermObj = Termimate_obj_fail

        # TERMINATIONS FOR MIMIC
        self.qpos_fail_thresh = 0.75
        self.TermPose = Termimate_pose_fail
        ##########################################

        self.object_bid = self.sim.model.body_name2id(self.object_name)
        # self.wrist_bid = self.sim.model.body_name2id("wrist")
        self.wrist_bid = self.sim.model.body_name2id("lunate")

        # turn off the body skeleton rendering
        self.sim.model.geom_rgba[self.sim.model.geom_name2id("body"), 3] = 0.0

        self._lift_z = self.sim.data.xipos[self.object_bid][2] + self.lift_bonus_thresh

        self.far_th = 0.001
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=10,
            sites=["IFtip"],
            **kwargs
        )

        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon  # doesn't work always. WIP

        # Adjust init as per the specified key
        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            robot_dim = self.ref.robot_dim
            self.init_qpos[:robot_dim] = robot_init

        if object_init is not None:
        # Make sure we have enough space in init_qpos
            if self.sim.model.nq >= (robot_dim + 6):  # 3 for position + 3 for orientation
                # Position
                self.init_qpos[robot_dim:robot_dim + 3] = object_init[:3]
                # Orientation (convert quaternion to euler)
                euler_angles = quat2euler(object_init[3:])
                self.init_qpos[robot_dim + 3:robot_dim + 6] = euler_angles
        else:
            print(f"Warning: model.nq ({self.sim.model.nq}) too small for full object state")

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True
        # if self.sim.model.nkey>0:
        # self.init_qpos[:] = self.sim.model.key_qpos[0,:]

    def rotation_distance(self, q1, q2, euler=True):
        if euler:
            q1 = euler2quat(q1)
            q2 = euler2quat(q2)

        return np.abs(quatDiff2Vel(q2, q1, 1)[0])

    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            self.sim.model.site_pos[self.target_sid][:] = curr_ref.object[:3]
            self.sim_obsd.model.site_pos[self.target_sid][:] = curr_ref.object[:3]
            
            self.sim.forward()

    def norm2(self, x):
        return np.sum(np.square(x))

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        print("target " + str(np.array(obs_dict['target_pos'])))
        print("tip_pos " + str(np.array(obs_dict['tip_pos'])))
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        near_th = len(self.tip_sids)*.0125

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist>far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def qpos_from_robot_object(self, qpos, robot, object):
        qpos[: len(robot)] = robot
        qpos[len(robot) : len(robot) + 3] = object[:3] 
        qpos[len(robot) + 3 :] = quat2euler(object[3:])

    def playback(self):
        idxs = self.ref.find_timeslot_in_reference(self.time + self.motion_start_time)
        # print(f"Time {self.time} {idxs} {self.ref.horizon}")
        ref_mot = self.ref.get_reference(self.time + self.motion_start_time)
        self.qpos_from_robot_object(self.sim.data.qpos, ref_mot.robot, ref_mot.object)
        self.sim.forward()
        self.sim.data.time = self.sim.data.time + 0.02  # self.env.env.dt
        return idxs[0] < self.ref.horizon - 1

    def reset(self, **kwargs):
        # print("Reset")
        self.ref.reset()
        obs = super().reset(
            reset_qpos=self.init_qpos, reset_qvel=self.init_qvel, **kwargs
        )

        return obs

    def check_termination(self, obs_dict):

        obj_term, qpos_term, base_term = False, False, False
        if self.TermObj:  # termination on object
            # object too far from reference
            obj_term = (
                True
                if self.norm2(obs_dict["obj_com_err"]) >= self.obj_fail_thresh**2
                else False
            )
            # wrist too far from object
            base_term = (
                True
                if self.norm2(obs_dict["base_error"]) >= self.base_fail_thresh**2
                else False
            )

        if self.TermPose:  # termination on posture
            qpos_term = (
                True
                if self.norm2(obs_dict["hand_qpos_err"]) >= self.qpos_fail_thresh
                else False
            )

        return (
            obj_term or qpos_term or base_term
        )  # combining termination for object + posture
