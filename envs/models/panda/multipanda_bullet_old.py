import os
import sys
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as Rot


class MultiPanda_bullet:
    def __init__(self, base_poses, base_orientations, stepsize=1e-3, realtime=0, GUI=False, debug=False, Collision=False, collision_shape='mesh', **kwargs):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque" 

        self.position_control_gain_p = [100.0] * 7
        self.position_control_gain_d = [40.0] * 7
        self.max_torque = [100] * 7
        
        self.GUI = GUI
        self.debug = debug
        
        self.name = 'MP'
        self.hand = False

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
        p.setGravity(0,0,0)

        # load models
        
        self.n_robot = len(base_poses)
        
        self.base_poses = base_poses
        self.base_orientations = base_orientations
        
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '..'))

        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane,-1,restitution=.95)

        self.robot_colors = ['red', 'blue', 'green', 'purple']
        
        self.robots = []
        self.dofs = []
        self.end_effector_indices = []
        
        self.joints = []
        self.q_min = []
        self.q_max = []
        self.qdot_max = []
        self.target_pos = []
        self.target_torque = []
        
        self.activedofs = []
        
        for r_idx in range(self.n_robot):
            ri_base_pos = self.base_poses[r_idx]
            ri_base_ori = p.getQuaternionFromEuler([0, 0, self.base_orientations[r_idx]])
            robot_color = self.robot_colors[r_idx%4]
            
            if collision_shape == 'capsule':
                self.robot_name = 'panda_capsule.urdf'
            else:
                self.robot_name = f'panda_{robot_color}.urdf'
                
            robot_i = p.loadURDF(f"panda/{self.robot_name}",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION,
                                basePosition = ri_base_pos,
                                baseOrientation = ri_base_ori)
            
            ri_dof = p.getNumJoints(robot_i)
            activedof = 0
            joints = []
            for joint_idx in range(ri_dof):
                tmp_joint_info = p.getJointInfo(robot_i, joint_idx)
                if tmp_joint_info[2] != 4: 
                    activedof += 1
                    joints.append(joint_idx)
                    self.q_min.append(tmp_joint_info[8])
                    self.q_max.append(tmp_joint_info[9])
                    self.qdot_max.append(tmp_joint_info[11])
                    self.target_pos.append((tmp_joint_info[8] + tmp_joint_info[9])/2.0)
                    self.target_torque.append(0.)
            
            self.robots.append(robot_i)
            self.dofs.append(ri_dof)
            self.activedofs.append(activedof)
            self.joints.append(joints)
            
        self.n_dof = np.array(self.activedofs, dtype=int).sum()
        
        self.range_angles = []
        # Add joint value slider
        if self.debug:
            for r_idx in range(self.n_robot):
                for j_idx in range(self.activedofs[r_idx]):
                    joint_idx = np.array(self.activedofs[:r_idx], dtype=int).sum() + j_idx
                    self.range_angles.append(p.addUserDebugParameter(f'R{r_idx+1}_J{j_idx+1}', 
                                                                rangeMin=self.q_min[joint_idx], 
                                                                rangeMax=self.q_max[joint_idx], 
                                                                startValue=self.target_pos[joint_idx]))
        
        if not Collision:
            for r1_idx in range(self.n_robot):
                for r2_idx in range(r1_idx+1, self.n_robot):
                    for l1 in self.joints[r1_idx]:
                        for l2 in self.joints[r2_idx]:
                            p.setCollisionFilterPair(self.robots[r1_idx], self.robots[r2_idx], l1, l2, 0)

        self.reset()
        
    def all2sep(self, joint_idx):
        for r_idx in range(self.n_robot):
            if joint_idx < np.array(self.activedofs[:r_idx+1], dtype=int).sum():
                break
        j_idx = joint_idx - np.array(self.activedofs[:r_idx], dtype=int).sum()
        return r_idx, self.joints[r_idx][j_idx]
    
    def sep2all(self, r_idx, j_idx):
        return np.array(self.activedofs[:r_idx], dtype=int).sum() + self.joints[r_idx].index(j_idx)

    def reset(self):
        self.t = 0.0        
        self.control_mode = "torque"
        for j in range(self.n_dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.0
            robot_idx, joint_idx = self.all2sep(j)
            p.resetJointState(self.robots[robot_idx], joint_idx, targetValue=self.target_pos[j])
            
        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()
    
    def resetController(self):
        for r_idx in range(self.n_robot):
            p.setJointMotorControlArray(bodyUniqueId=self.robots[r_idx],
                                        jointIndices=self.joints[r_idx],
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0. for i in range(self.activedofs[r_idx])])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')
        
    def reset2TargetPositions(self, target_pos):
        assert len(target_pos) == self.n_dof
        self.target_pos = target_pos
        for joint_idx in range(self.n_dof):
            r_idx, j_idx = self.all2sep(joint_idx)
            p.resetJointState(self.robots[r_idx], j_idx, target_pos[joint_idx])
    
    def getJointStates(self):
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        
        for joint_idx in range(self.n_dof):
            r_idx, j_idx = self.all2sep(joint_idx)
            
            joint_state = p.getJointStates(self.robots[r_idx], j_idx)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
            joint_torques.append(joint_state[3])
            
        return joint_positions, joint_velocities, joint_torques
    
    def setTargetPositions(self, target_pos):
        raise NotImplementedError
        
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
        if len(list(set(key_codes) & set(arrow_codes))) != 0:
            if list(key_codes)[0] == 65297:  # UP
                camera_pos = list(np.array(camera_pos) + np.array(camera_up) * 0.1)
            if list(key_codes)[0] == 65298:  # DOWN
                camera_pos = list(np.array(camera_pos) + np.array(camera_down) * 0.1)
            if list(key_codes)[0] == 65295:  # LEFT
                camera_pos = list(np.array(camera_pos) + np.array(camera_left) * 0.1)
            if list(key_codes)[0] == 65296:  # RIGHT
                camera_pos = list(np.array(camera_pos) + np.array(camera_right) * 0.1)

            p.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                        cameraYaw=camera_yaw,
                                        cameraPitch=camera_pitch,
                                        cameraTargetPosition=camera_pos)
            
    def check_collision(self, pos, check_threshold=2, return_links=False):
        pos = np.asarray(pos)
        assert len(pos.shape) == 1, 'Only for single joint value'

        self.reset2TargetPositions(pos)
        p.stepSimulation()
        
        closest_points = []
        for r1_idx in range(self.n_robot):
            for r2_idx in range(r1_idx+1, self.n_robot):
                closest_points += list(p.getClosestPoints(bodyA = self.robots[r1_idx], bodyB = self.robots[r2_idx], distance = check_threshold))
                
        min_dist = 1e10
        col_links = [-100, -100]
        
        for closest_point in closest_points:
            if min_dist > closest_point[8]:
                min_dist = closest_point[8]
                col_links = closest_point[3:5]
        
        if return_links:
            return min_dist, col_links
        else:
            return min_dist
        
    def get_distance_between_links(self, l1_idx, l2_idx):
        
        # base link idx : -1 * r_idx (ex. base link of robot 3 == -3)
        
        if l1_idx < 0:
            l1_r_idx = int(l1_idx / (-1))-1
            l1_l_idx = -1
        else:
            l1_r_idx, l1_l_idx = self.all2sep(l1_idx)
        
        if l2_idx < 0:
            l2_r_idx = int(l2_idx / (-1))-1
            l2_l_idx = -1
        else:
            l2_r_idx, l2_l_idx = self.all2sep(l2_idx)
        
        closest_points = p.getClosestPoints(
            bodyA = self.robots[l1_r_idx], 
            linkIndexA=l1_l_idx,
            bodyB = self.robots[l2_r_idx], 
            linkIndexB=l2_l_idx,
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
    
    base_poses = [[0, 0.5, 0], [0, -0.5, 0], [0.5, 0, 0], [-0.5, 0, 0]]
    base_orientations = [-1.5708, 1.5708, 3.1415, 0]
    
    env = MultiPanda_bullet(base_poses, base_orientations, debug=True, GUI=True, Collision=False, collision_shape='mesh')
    
    while True:
        pos = env.getTargetPositionfromGUI()
        env.reset2TargetPositions(pos)
        dist = env.check_collision(pos)
        print(f'Min. dist. {dist:.3f}m')
        
        # print(env.get_distance_between_links(6, -3))
        
        env.setView()