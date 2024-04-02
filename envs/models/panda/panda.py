import torch
import open3d as o3d

import os
import fcl
import hppfcl
from scipy.spatial.transform import Rotation as Rot

from envs.lib.LieGroup import *

class Panda:
    def __init__(self, T_base=np.eye(4), T_ee=np.eye(4), hand=False, finger=False, device='cpu', collision_shape='mesh', mesh_type='simplified'):
        
        self.n_dof = 7
        self.T_base = torch.as_tensor(T_base, dtype=torch.float).to(device)
        self.hand = hand
        self.finger = finger
        
        # screws A_i, i-th screw described in i-th frame
        self.A = torch.tensor([ [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [1, 1, 1, 1, 1, 1, 1], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0.0]]).to(device)
        
        # M : M_i == link frames M_{i,i-1}
        self.M = torch.zeros((4, 4, self.n_dof))
        self.M[:, :, 0] = torch.tensor([[1, 0, 0, 0], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 1, 0.333], 
                                        [0, 0, 0, 1]])
        
        self.M[:, :, 1] = torch.tensor([[1, 0, 0, 0], 
                                        [0, 0, 1, 0], 
                                        [0,-1, 0, 0], 
                                        [0, 0, 0, 1.0]])
        
        self.M[:, :, 2] = torch.tensor([[1, 0, 0, 0], 
                                        [0, 0, -1, -0.316], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 0, 1]])
        
        self.M[:, :, 3] = torch.tensor([[1, 0, 0, 0.0825], 
                                        [0, 0,-1, 0], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 0, 1]])
        
        self.M[:, :, 4] = torch.tensor([[1, 0, 0, -0.0825], 
                                        [0, 0, 1, 0.384], 
                                        [0,-1, 0, 0], 
                                        [0, 0, 0, 1]])
            
        self.M[:, :, 5] = torch.tensor([[1, 0, 0, 0], 
                                        [0, 0,-1, 0], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 0, 1.0]])
        
        self.M[:, :, 6] = torch.tensor([[1, 0, 0, 0.088], 
                                        [0, 0,-1, 0], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 0, 1]])
        
        for i in range(self.n_dof):
            self.M[:, :, i] = invSE3(self.M[:, :, i].unsqueeze(0)).squeeze(0)
        self.M = self.M.to(device)
        
        if self.hand:
            T_hand = torch.eye(4, dtype=torch.float)
            T_hand[:3, :3] = torch.as_tensor(Rot.from_euler('XYZ', [0.0, 0.0, -0.7854]).as_matrix())
            T_hand[:3, 3] = torch.as_tensor([0, 0, 0.107])
            self.M_hand =  T_hand.to(device)
            if self.finger:
                raise NotImplementedError
            else:
                T_finger = torch.eye(4, dtype=torch.float).to(device)
                T_finger[:3, 3] = torch.as_tensor([0.0, 0.0, 0.09])
                self.M_finger = T_finger.to(device)
                self.M_ee = self.M_hand @ self.M_finger @ torch.as_tensor(T_ee, dtype=torch.float).to(device)
        else:
            self.M_ee = torch.as_tensor(T_ee, dtype=torch.float).to(device)
        
        self.M_sb = torch.eye(4).to(device)
        for i in range(self.n_dof):
            self.M_sb = self.M_sb @ invSE3(self.M[:, :, i].unsqueeze(0)).squeeze(0)
        self.M_sb = self.M_sb @ self.M_ee
        
        # Spatial Screw A_s and Body Screw A_b
        self.A_s = torch.zeros(self.A.shape).to(device)
        self.A_b = torch.zeros(self.A.shape).to(device)
        M_si = torch.eye(4).to(device)
        
        for i in range(self.n_dof):
            M_si = M_si @ invSE3(self.M[:, :, i].unsqueeze(0)).squeeze(0)
            self.A_s[:, i] = largeAdjoint(M_si.unsqueeze(0)).squeeze(0) @ self.A[:, i]
            self.A_b[:, i] = largeAdjoint(invSE3(self.M_sb.unsqueeze(0))).squeeze(0) @ self.A_s[:, i]
            
        self.q_min = torch.tensor([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]]).T.to(device)
        self.q_max = torch.tensor([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.752, 2.8973]]).T.to(device)
        self.qdot_min = torch.tensor([[-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]]).T.to(device)
        self.qdot_max = torch.tensor([[2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]]).T.to(device)
        
        if mesh_type == 'original':
            MESH_PATH = './envs/models/panda/meshes/visual'
        elif mesh_type == 'simplified':
            MESH_PATH = './envs/models/panda/meshes/collision'
        else:
            raise NotImplementedError
        
        self.meshes = [o3d.io.read_triangle_mesh(os.path.join(MESH_PATH, f'link{idx}.stl')) for idx in range(self.n_dof+1)]
        self.vertices = [torch.tensor(np.asarray(mesh.vertices), dtype=torch.float).to(device) for mesh in self.meshes]
        self.triangles = [torch.tensor(np.asarray(mesh.triangles), dtype=torch.int).to(device) for mesh in self.meshes]
        
        if self.hand:
            self.meshes.append(o3d.io.read_triangle_mesh(os.path.join(MESH_PATH, f'hand.stl')))
            self.vertices.append(torch.tensor(np.asarray(self.meshes[-1].vertices), dtype=torch.float).to(device))
            self.triangles.append(torch.tensor(np.asarray(self.meshes[-1].triangles), dtype=torch.int).to(device))
            
            if self.finger:
                raise NotImplementedError
            else:
                self.meshes.append(o3d.io.read_triangle_mesh(os.path.join(MESH_PATH, f'panda_fingerbox.obj')))
                self.vertices.append(torch.tensor(np.asarray(self.meshes[-1].vertices), dtype=torch.float).to(device))
                self.triangles.append(torch.tensor(np.asarray(self.meshes[-1].triangles), dtype=torch.int).to(device))
        
        self.collision_shape = collision_shape
        self.fclCollisionObjects = []
        self.hppfclCollisionObjects = []
        
        if self.collision_shape == 'mesh':
            for idx in range(len(self.vertices)):
                tmpV = self.vertices[idx]
                tmpT = self.triangles[idx]
                tmpshape = fcl.BVHModel()
                tmpshape.beginModel(len(tmpV), len(tmpT))
                tmpshape.addSubModel(tmpV.cpu().numpy(), tmpT.cpu().numpy())
                tmpshape.endModel()
                tmpobj = fcl.CollisionObject(tmpshape, fcl.Transform())
                self.fclCollisionObjects.append(tmpobj)
                
                V_hpp = hppfcl.StdVec_Vec3f()
                T_hpp = hppfcl.StdVec_Triangle()
                V_hpp.extend([tmpV[i].cpu().numpy() for i in range(len(tmpV))])
                for idx in range(len(tmpT)):
                    T_hpp.append(hppfcl.Triangle(tmpT[idx, 0].item(), tmpT[idx, 1].item(), tmpT[idx, 2].item()))
                
                hppobj = hppfcl.CollisionObject(hppfcl.Convex(V_hpp, T_hpp))
                hppobj.setTransform(hppfcl.Transform3f.Identity())
                self.hppfclCollisionObjects.append(hppobj)
        
        elif self.collision_shape == 'capsule':
            
            assert not self.hand, 'Hand is not supported for capsule collision shape'
            
            p1_list = [
                [-0.05465271, -0.00160026,  0.01809338],
                [-0.00020431,  0.00631866, -0.14863947],
                [-0.00611539, -0.14765818,  0.00210107],
                [ 0.07971434,  0.04523061, -0.00080541],
                [-0.08218869,  0.08021483,  0.00232922],
                [-1.5345939e-04,  8.8221747e-03, -2.3295735e-01],
                [0.08474172, 0.00705573, 0.0045368 ],
                [0.03195187, 0.03167332, 0.06886173]
            ]
            p2_list = [
                [-3.6190036e-03,  5.1440922e-05,  5.1162861e-02],
                [ 0.00130181, -0.05217966, -0.00339115],
                [ 0.0010172,  -0.0026326,   0.05261712],
                [ 0.00334147, -0.00450623, -0.07987815],
                [-0.00809085,  0.00193277,  0.04148006],
                [-0.00389436,  0.0633805,  -0.00224684],
                [ 2.5954140e-02, -1.0050392e-03, -2.5985579e-05],
                [-0.00688292, -0.00610242,  0.07171543]
            ]
            r_list = [
                0.10731089115142822,
                0.07845213264226913,
                0.07795113325119019,
                0.0734412893652916,
                0.07551460713148117,
                0.0707956999540329,
                0.07869726419448853,
                0.05470610037446022
            ]
            p1s = torch.tensor(p1_list)
            p2s = torch.tensor(p2_list)
            rs = torch.tensor(r_list)
            
            self.T_capsule = torch.eye(4, device=device).unsqueeze(0).repeat_interleave(self.n_dof+1, dim=0)
            
            for idx, (p1, p2, r) in enumerate(zip(p1s, p2s, rs)):
                h = torch.norm(p1-p2)
                tmp_geom = fcl.Capsule(r, h)
                
                v = p2-p1
                w = torch.cross(torch.tensor([0.0, 0.0, 1.0]), v)
                w = w / torch.norm(w)
                
                theta = torch.arccos(torch.dot(torch.tensor([0.0, 0.0, 1.0]), v) / torch.norm(v))
                T_link = torch.eye(4)
                T_link[:3, :3] = torch.matrix_exp(skew_so3((w*theta).unsqueeze(0)).squeeze(0))
                T_link[:3, 3] = p1 + T_link[:3, :3] @ torch.tensor([0.0, 0.0, h/2])
                
                self.T_capsule[idx] = T_link.to(self.T_capsule)
                
                self.fclCollisionObjects.append(fcl.CollisionObject(tmp_geom, fcl.Transform()))
        
        self.device = device 
        
    def solveForwardKinematics(self, jointPos, return_T_link=False):
        
        jointPos = torch.as_tensor(jointPos, dtype=torch.float).to(self.device)
            
        M_ = torch.zeros((self.n_dof, 4, 4)).to(self.device)
        for i in range(self.n_dof):
            M_[i, :, :] = invSE3(self.M[:, :, i].unsqueeze(0)).squeeze(0)
        
        M_sb = torch.eye(4).to(self.device)
        T_sj = torch.zeros(self.n_dof, 4, 4).to(self.device)
        for i in range(self.n_dof):
            M_sb = M_sb @ M_[i, :, :] @ expSE3((self.A[:, i]*jointPos[i]).unsqueeze(0)).squeeze(0)
            T_sj[i] = self.T_base @ M_sb
        
        T_sb = self.T_base @ M_sb @ self.M_ee
        if self.hand:
            T_hand = self.T_base @ M_sb @ self.M_hand
            T_finger = self.T_base @ M_sb @ self.M_hand @ self.M_finger
            T_sj = torch.cat([T_sj, T_hand.unsqueeze(0), T_finger.unsqueeze(0)], dim=0)
        
        if return_T_link:
            return T_sb, T_sj
        else:
            return T_sb
        
    def solveBatchForwardKinematics(self, jointPos, return_T_link=False):
        if not isinstance(jointPos, torch.Tensor):
            jointPos = torch.tensor(jointPos)
        jointPos = jointPos.to(self.device)
        
        assert len(jointPos.shape) == 2 and jointPos.shape[1] == self.n_dof
        
        B = len(jointPos)
        
        M_ = torch.zeros((self.n_dof, 4, 4)).to(self.device)
        for i in range(self.n_dof):
            M_[i, :, :] = invSE3(self.M[:, :, i].unsqueeze(0)).squeeze(0)
            
        M_ = M_.unsqueeze(0).repeat_interleave(B, dim=0)
        
        A_ = self.A.unsqueeze(0).repeat_interleave(B, dim=0)
        T_base_ = self.T_base.unsqueeze(0).repeat_interleave(B, dim=0)
        M_ee_ = self.M_ee.unsqueeze(0).repeat_interleave(B, dim=0)
        
        M_sb = torch.eye(4).unsqueeze(0).repeat_interleave(B, dim=0).to(self.device)
        T_sj = torch.zeros(B, self.n_dof, 4, 4).to(self.device)
        
        A_[:, :, i]*jointPos[:, i].unsqueeze(1).repeat_interleave(6, dim=1)
        for i in range(self.n_dof):
            M_tmp = expSE3(A_[:, :, i]*jointPos[:, i].unsqueeze(1).repeat_interleave(6, dim=1))
            M_sb = M_sb @ M_[:, i, :, :] @ M_tmp
            # (b, 4, 4) @ (b, 4, 4) @ expSE3((b, 6) * (b, 6))
            T_sj[:, i] = T_base_ @ M_sb
            
        T_sb = T_base_ @ M_sb @ M_ee_
        
        if self.hand:
            M_hand_ = self.M_hand.unsqueeze(0).repeat_interleave(B, dim=0)
            M_finger_ = self.M_finger.unsqueeze(0).repeat_interleave(B, dim=0)
            
            T_hand = T_base_ @ M_sb @ M_hand_
            T_finger = T_base_ @ M_sb @ M_hand_ @ M_finger_
            T_sj = torch.cat([T_sj, T_hand.unsqueeze(1), T_finger.unsqueeze(1)], dim=1)
            
        if return_T_link:
            return T_sb, T_sj
        else:
            return T_sb
    
    def get_meshes(self, jointPos):
        _, T_sj = self.solveForwardKinematics(jointPos, return_T_link=True)
        
        vertices = []
        triangles = []
        
        V = self.vertices[0]
        V = (self.T_base[:3, :3] @ V.T + self.T_base[:3, 3].unsqueeze(-1).repeat(1, len(V))).T
        vertices.append(V)
        triangles.append(self.triangles[0])
        
        n_vertices = len(self.vertices[0])
        
        for i in range(1, len(self.vertices)):
            V = self.vertices[i]
            V = (T_sj[i-1, :3, :3] @ V.T + T_sj[i-1, :3, 3].unsqueeze(-1).repeat(1, len(V))).T

            T = self.triangles[i] + n_vertices
            n_vertices += len(V)

            vertices.append(V)
            triangles.append(T)
            
        vertices = torch.cat(vertices, dim=0)
        triangles = torch.cat(triangles, dim=0)
        
        return vertices, triangles
    
    def fcl_objs(self, jointPos, **kwargs):
        _, T_sj = self.solveForwardKinematics(jointPos, return_T_link=True)

        type = kwargs.get('type', self.collision_shape)
        
        if type == 'mesh':
            
            for idx in range(len(self.fclCollisionObjects)):
                if idx == 0:
                    Transform = self.T_base
                else:
                    Transform = T_sj[idx-1]
                    
                tmp_transform = fcl.Transform(Transform[:3, :3].cpu(), Transform[:3, 3].cpu())
                
                self.fclCollisionObjects[idx].setTransform(tmp_transform)
                
        elif type == 'capsule':
                
            for idx in range(len(self.fclCollisionObjects)):
                if idx == 0:
                    Transform = self.T_base @ self.T_capsule[idx]
                else:
                    Transform = T_sj[idx-1] @ self.T_capsule[idx]
                    
                tmp_transform = fcl.Transform(Transform[:3, :3].cpu(), Transform[:3, 3].cpu())
                
                self.fclCollisionObjects[idx].setTransform(tmp_transform)
                
        return self.fclCollisionObjects
    
    def hppfcl_objs(self, jointPos, **kwargs):
        _, T_sj = self.solveForwardKinematics(jointPos, return_T_link=True)
        
        type = kwargs.get('type', self.collision_shape)
        
        if type == 'mesh':
            
            for idx in range(len(self.hppfclCollisionObjects)):
                if idx == 0:
                    Transform = self.T_base
                else:
                    Transform = T_sj[idx-1]
                    
                tmp_transform = hppfcl.Transform3f.Identity()
                tmp_transform.setRotation(Transform[:3, :3].cpu().numpy())
                tmp_transform.setTranslation(Transform[:3, 3].cpu().numpy())
                
                self.hppfclCollisionObjects[idx].setTransform(tmp_transform)
                
        elif type == 'capsule':
                
            for idx in range(len(self.hppfclCollisionObjects)):
                if idx == 0:
                    Transform = self.T_base @ self.T_capsule[idx]
                else:
                    Transform = T_sj[idx-1] @ self.T_capsule[idx]
                    
                tmp_transform = hppfcl.Transform3f.Identity()
                tmp_transform.setRotation(Transform[:3, :3].cpu().numpy())
                tmp_transform.setTranslation(Transform[:3, 3].cpu().numpy())
                
                self.hppfclCollisionObjects[idx].setTransform(tmp_transform)

        return self.hppfclCollisionObjects