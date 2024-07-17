import os
import sys
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as Rot


class MultiPanda_bullet:
    def __init__(self, 
            base_poses,                 # Robot Arm Base Positions : List of [x, y, z]
            base_orientations,          # Robot Arm Base Orientations : List of Z-axis angle
            obstacles=None,             # Obstacle config dictionary
            stepsize=1e-3, 
            realtime=0, 
            hand=False,                 # Franka Emika Panda with hand: Boolean
            finger=False,               # Finger: Boolean (True: w/ finger mesh, False: w/ finger box)
            Collision=False,            # Collision in the simulation : Boolean
            collision_shape='mesh',     # Collision shape of the robot arm : 'mesh' or 'capsule'
            GUI=False,                  # GUI mode : Boolean
            debug=False,                # Debug mode (joint value slider) : Boolean
            **kwargs):
        
        self.stepsize = stepsize
        self.realtime = realtime
        
        self.GUI = GUI
        self.debug = debug
        self.hand = hand
        self.finger = finger

        # connect pybullet
        if self.GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, 0)

        # load models
        
        self.n_robot = len(base_poses)
        self.base_poses = base_poses
        self.base_orientations = base_orientations
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles
        
        self.n_objects = 0
        
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '..'))

        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)

        self.robot_colors = [[1.0, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0], [0.5, 1.0, 0.5, 1.0], [1.0, 0.5, 1.0, 1.0]]
        
        self.robots = []
        self.dofs = []
        self.end_effector_indices = []
        
        self.joints = []
        self.q_min = []
        self.q_max = []
        self.qdot_max = []
        self.target_pos = []
        
        self.activedofs = []
        
        self.bodies = []
        self.body_dict = {}
        
        for r_idx in range(self.n_robot):
            ri_base_pos = self.base_poses[r_idx]
            ri_base_ori = p.getQuaternionFromEuler([0, 0, self.base_orientations[r_idx]])
            robot_color = self.robot_colors[r_idx%4]
            
            if collision_shape == 'capsule':
                assert not self.hand, 'Capsule collision shape is not supported for hand'
                self.robot_name = 'panda_capsule.urdf'
            else:
                if self.hand:
                    self.robot_name = f'panda_handbox.urdf'
                else:
                    self.robot_name = f'panda.urdf'
                
            robot_i = p.loadURDF(
                f"panda/{self.robot_name}",
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
                basePosition = ri_base_pos,
                baseOrientation = ri_base_ori)
            
            ri_dof = p.getNumJoints(robot_i)
            activedof = 0
            joints = []
            links = [-1]
            for joint_idx in range(ri_dof):
                p.changeVisualShape(robot_i, joint_idx, rgbaColor=robot_color)
                links.append(joint_idx)
                tmp_joint_info = p.getJointInfo(robot_i, joint_idx)
                if tmp_joint_info[2] != 4: # ignore fixed joint
                    activedof += 1
                    joints.append(joint_idx)
                    self.q_min.append(tmp_joint_info[8])
                    self.q_max.append(tmp_joint_info[9])
                    self.qdot_max.append(tmp_joint_info[11])
                    self.target_pos.append((tmp_joint_info[8] + tmp_joint_info[9])/2.0)
            
            self.robots.append(robot_i)
            self.dofs.append(ri_dof)
            self.activedofs.append(activedof)
            self.joints.append(joints)
            self.n_objects += len(links)
            
            body_info = {
                'name': f'robot_{r_idx}',
                'type': 'robot',
                'id': robot_i,
                'links': links,
                'n_links': len(links),
                'position': ri_base_pos,
                'orientation': ri_base_ori,
                
                'joints': joints,
                'n_dof': len(joints),
            }
            self.bodies.append(body_info)
            self.body_dict[robot_i] = body_info
            
        self.n_dof = np.array(self.activedofs, dtype=int).sum()
        
        # obstacles
        for obs_idx, obstacle in enumerate(self.obstacles):
            name = obstacle['name']
            position = obstacle['position']
            orientation = obstacle['orientation']
            
            if type(orientation) in [float, int]:
                orientation = p.getQuaternionFromEuler([0, 0, orientation])
                obstacle['orientation'] = orientation
                
            obstacle['id'] = p.loadURDF(f'obstacles/{name}.urdf', 
                                        basePosition=position, 
                                        baseOrientation=orientation,
                                        useFixedBase=True)
            
            obs_links = [-1] + list(range(p.getNumJoints(obstacle['id'])))
            self.n_objects += len(obs_links)
            
            body_info = {
                'name': f'{name}_{obs_idx}',
                'type': 'obstacle',
                'id': obstacle['id'],
                'links': obs_links,
                'n_links': len(obs_links),
                'position': position,
                'orientation': orientation,
            }
            self.bodies.append(body_info)
            self.body_dict[obstacle['id']] = body_info
            
        # Add joint value slider
        self.range_angles = []
        if self.debug:
            for r_idx in range(self.n_robot):
                for j_idx in range(self.activedofs[r_idx]):
                    joint_idx = np.array(self.activedofs[:r_idx], dtype=int).sum() + j_idx
                    self.range_angles.append(p.addUserDebugParameter(f'R{r_idx+1}_J{j_idx+1}', 
                                                                rangeMin=self.q_min[joint_idx], 
                                                                rangeMax=self.q_max[joint_idx], 
                                                                startValue=self.target_pos[joint_idx]))
        
        if not Collision:
            for o1_idx in range(self.n_objects):
                for o2_idx in range(o1_idx+1, self.n_objects):
                    o1_id, o1_link = self.idx2id(o1_idx)
                    o2_id, o2_link = self.idx2id(o2_idx)
                    p.setCollisionFilterPair(o1_id, o2_id, o1_link, o2_link, 0)

        self.collision_pairs = []
        self_collision_pairs = [
            {-1, 4}, {-1, 5}, {-1, 6}, {-1, 7}, {-1, 8},            # base ~ fingerbox : -1 ~ 8
            {0, 5}, {0, 6}, {0, 7}, {0, 8}, {1, 7}, {1, 8}, 
        ]
        for o1_idx in range(self.n_objects):
            for o2_idx in range(o1_idx+1, self.n_objects):
                o1_bID, o1_lID = self.idx2id(o1_idx)
                o2_bID, o2_lID = self.idx2id(o2_idx)
                o1_info = self.body_dict[o1_bID]
                o2_info = self.body_dict[o2_bID]
                
                # ignore distances between obstacles
                if o1_info['type'] == 'obstacle' and o2_info['type'] == 'obstacle':
                    continue
                # ignore distances between robot bases and obstacles
                elif (o1_info['type'] == 'robot' and o1_lID == -1 and o2_info['type'] == 'obstacle') or (o2_info['type'] == 'robot' and o2_lID == -1 and o1_info['type'] == 'obstacle'):
                    continue
                # self collision
                elif o1_bID == o2_bID and o1_info['type'] == 'robot':
                    if {o1_lID, o2_lID} in self_collision_pairs:
                        self.collision_pairs.append([o1_idx, o2_idx])
                else:
                    self.collision_pairs.append([o1_idx, o2_idx])

        # Control Gains
        self.position_control_gain_p = [100.0] * self.n_dof
        self.position_control_gain_d = [40.0] * self.n_dof
        self.max_torque = [100] * self.n_dof
        
        self.reset()
        
    def idx2id_joint(self, joint_idx):
        # total joint index to robot and joint id
        assert joint_idx in list(range(self.n_dof)), f'invalid index of the joint. possible: 0 ~ {self.n_dof-1}, got {joint_idx}'
        
        for body_info in self.bodies:
            if body_info['type'] == 'robot':
                if joint_idx < body_info['n_dof']:
                    break
                else:
                    joint_idx = joint_idx - body_info['n_dof']
        
        return body_info['id'], body_info['joints'][joint_idx]
    
    def idx2id(self, obj_idx):
        
        assert obj_idx in list(range(self.n_objects)), f'invalid index of the object. possible: 0 ~ {self.n_objects-1}, got {obj_idx}'
        
        for body_info in self.bodies:
            if obj_idx < body_info['n_links']:
                return body_info['id'], body_info['links'][obj_idx]
            else:
                obj_idx = obj_idx - body_info['n_links']
    
    def id2idx(self, bodyID, linkID):
        is_bodyiID_exist = False
        obj_idx = 0
        for body_idx, body_info in enumerate(self.bodies):
            if bodyID == body_info['id']:
                is_bodyiID_exist = True
                break
            else:
                obj_idx += body_info['n_links']
                
        assert is_bodyiID_exist, f'invalid bodyID: {bodyID}'
        assert linkID in body_info['links'], f'invalid linkID: {linkID}'
        
        obj_idx += body_info['links'].index(linkID)
        return obj_idx

    def reset(self):    
        for j in range(self.n_dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            robot_id, joint_id = self.idx2id_joint(j)
            p.resetJointState(robot_id, joint_id, targetValue=self.target_pos[j])

    def step(self):
        p.stepSimulation()
        
    def reset2TargetPositions(self, target_pos):
        assert len(target_pos) == self.n_dof
        self.target_pos = target_pos
        for joint_idx in range(self.n_dof):
            r_id, j_id = self.idx2id_joint(joint_idx)
            p.resetJointState(r_id, j_id, target_pos[joint_idx])
            
    def set2TargetPositions(self, target_pos):
        assert len(target_pos) == self.n_dof
        self.target_pos = target_pos
        for joint_idx in range(self.n_dof):
            r_id, j_id = self.idx2id_joint(joint_idx)
            p.setJointMotorControl2(
                bodyUniqueId=r_id,
                jointIndex=j_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos[joint_idx],
                force=self.max_torque[joint_idx],
                positionGain=self.position_control_gain_p[joint_idx],
                velocityGain=self.position_control_gain_d[joint_idx],
            )
            
    def set2TargetVelocities(self, target_vel):
        assert len(target_vel) == self.n_dof
        self.target_vel = target_vel
        for joint_idx in range(self.n_dof):
            r_id, j_id = self.idx2id_joint(joint_idx)
            p.setJointMotorControl2(
                bodyUniqueId=r_id,
                jointIndex=j_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel[joint_idx],
                force=self.max_torque[joint_idx],
            )
            
    def set2TargetTorques(self, target_torque):
        assert len(target_torque) == self.n_dof
        self.target_torque = target_torque
        for joint_idx in range(self.n_dof):
            r_id, j_id = self.idx2id_joint(joint_idx)
            p.setJointMotorControl2(
                bodyUniqueId=r_id,
                jointIndex=j_id,
                controlMode=p.TORQUE_CONTROL,
                force=target_torque[joint_idx],
            )
            
    def getJointStates(self):
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        
        for joint_idx in range(self.n_dof):
            r_id, j_id = self.idx2id_joint(joint_idx)
            joint_state = p.getJointStates(r_id, [j_id])[0]
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
            joint_torques.append(joint_state[3])
            
        return joint_positions, joint_velocities, joint_torques
        
    def getTargetPositionfromGUI(self):
        assert self.GUI and self.debug, "This function is only for GUI+debug mode."
        GUI_position = []
        for param_id in self.range_angles:
            GUI_position.append(p.readUserDebugParameter(param_id))
        return GUI_position
            
    def setView(self):
        events = p.getKeyboardEvents()
        key_codes = events.keys()

        camera_info = p.getDebugVisualizerCamera()
        camera_upaxis = camera_info[4]
        camera_forwardaxis = camera_info[5]
        camera_pos = camera_info[11]
        camera_dist = camera_info[10]
        camera_yaw = camera_info[8]
        camera_pitch = camera_info[9]

        camera_left = np.cross(camera_upaxis, camera_forwardaxis)
        camera_right = np.cross(camera_forwardaxis, camera_upaxis)
        camera_down = np.cross(camera_left, camera_forwardaxis)
        camera_up = np.cross(camera_right, camera_forwardaxis)

        arrow_codes = [65297, 65298, 65295, 65296]
        moving_vel = 0.05
        if len(list(set(key_codes) & set(arrow_codes))) != 0:
            if list(key_codes)[0] == 65297:  # UP
                camera_pos = list(np.array(camera_pos) + np.array(camera_up) * moving_vel)
            if list(key_codes)[0] == 65298:  # DOWN
                camera_pos = list(np.array(camera_pos) + np.array(camera_down) * moving_vel)
            if list(key_codes)[0] == 65295:  # LEFT
                camera_pos = list(np.array(camera_pos) + np.array(camera_left) * moving_vel)
            if list(key_codes)[0] == 65296:  # RIGHT
                camera_pos = list(np.array(camera_pos) + np.array(camera_right) * moving_vel)

            p.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                        cameraYaw=camera_yaw,
                                        cameraPitch=camera_pitch,
                                        cameraTargetPosition=camera_pos)
            
    def check_collision(self, pos, check_threshold=2, return_points=False):
        pos = np.asarray(pos)
        assert len(pos.shape) == 1, 'Only for single joint value'

        self.reset2TargetPositions(pos)
        p.stepSimulation()
        
        closest_points = []
        
        for pair in self.collision_pairs:
            o1_bid, o1_lid = self.idx2id(pair[0])
            o2_bid, o2_lid = self.idx2id(pair[1])
            closest_points += list(p.getClosestPoints(
                bodyA=o1_bid, 
                linkIndexA=o1_lid,
                bodyB=o2_bid, 
                linkIndexB=o2_lid,
                distance=check_threshold
            ))

        min_dist = 1e10
        col_links = [-100, -100]
        
        for closest_point in closest_points:
            if min_dist > closest_point[8]:
                min_dist = closest_point[8]
                points = closest_point[5:7]
        
        if return_points:
            return min_dist, points
        else:
            return min_dist
        
    def get_distance_between_objects(self, o1_idx, o2_idx):
        
        o1_bID, o1_lID = self.idx2id(o1_idx)
        o2_bID, o2_lID = self.idx2id(o2_idx)
        
        closest_points = p.getClosestPoints(
            bodyA = o1_bID, 
            linkIndexA=o1_lID,
            bodyB = o2_bID, 
            linkIndexB=o2_lID,
            distance=5)
        
        min_dist = closest_points[0][8]
        
        return min_dist
        
    def get_image(self, pos, width=1920, height=1080, roll=0, pitch=-20, yaw=45, camdistance=3.5, camTargetPos = [0, 0, 0.4], fov = 20):
        
        self.reset2TargetPositions(pos)
        p.stepSimulation()
        upAxisIndex = 2
        nearPlane = 0.01
        farPlane = 100
        
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camdistance, yaw, pitch, roll, upAxisIndex)
        aspect = width / height
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        img_arr = p.getCameraImage(width, height, viewMatrix, projectionMatrix)
        
        return img_arr[2]
    
if __name__ == '__main__':
    
    # base_poses = [[0, 0.5, 0], [0, -0.5, 0], [0.5, 0, 0], [-0.5, 0, 0]]
    # base_orientations = [-1.5708, 1.5708, 3.1415, 0]
    
    base_poses = [[0, 0, 0], [1.2, 0, 0]]
    base_orientations = [0, 3.1415]
    obstacles = [
        
        {
            'name': 'high_table',
            'position': [0.6, 0.0, 0.22132],
            'orientation': [0, 0, 0, 1]
        },
        
        # {
        #     'name': 'low_table',
        #     'position': [0.26, 0, -0.075/2-0.012],
        #     'orientation': 0.0
        # },
        # {
        #     'name': 'high_table',
        #     'position': [0.89815, 0.20346, 0.22132],
        #     'orientation': [-0.01028, -0.00131, -0.01,  0.9999]
        # },
        # {
        #     'name': 'shelve',
        #     'position': [0.10287, 0.8804, 0.00539],
        #     'orientation': [-0.01252, 0.00214, 0.70997, -0.70411]
        # },
    ]
    
    env = MultiPanda_bullet(
        base_poses, 
        base_orientations, 
        obstacles=obstacles, 
        hand=True, 
        finger=False, 
        Collision=False, 
        collision_shape='mesh',
        GUI=True, 
        debug=True, 
        )
    
    while True:
        pos = env.getTargetPositionfromGUI()
        env.reset2TargetPositions(pos)
        t = time.time()
        dist, points = env.check_collision(pos, return_points=True)
        t = time.time() - t
        print(f'Min. dist. {dist:.3f}m, elapsed time: {t:.3f}s')
        
        p.addUserDebugLine(points[0], points[1], [1, 0, 0], 5, 0.1)