import torch
import gc, sys, os

from tqdm import tqdm, trange

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from utils import now

class Node:
    def __init__(self, p, parent=None):
        self.p = torch.as_tensor(p, dtype=torch.float)
        self.parent = parent
        self.childs = []
        self.onpath = False
        
    def connect(self, child_node):
        child_node.parent = self
        self.childs.append(child_node)
    
    def to(self, device):
        self.p = self.p.to(device)
        return self

class RRTplanar:
    def __init__(self, step_size, q_min, q_max, device, goal_sample_rate=0.05, max_iter=10000):
        self.step_size = step_size
        self.q_min = torch.as_tensor(q_min).to(device)
        self.q_max = torch.as_tensor(q_max).to(device)
        self.dim = len(q_min)
        self.nodes = []
        
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.planned = False
        self.device = device
    
    def planning(self, start, end, col_checker, pbar=False):
        gc.disable()
        assert len(start) == len(end) == self.dim, f'expected dim. of input is {self.dim}, got {len(start)}'
        self.planned = False
        
        start_node = Node(start).to(self.device)
        end_node = Node(end).to(self.device)
        self.nodes.append(start_node)
        self.goal_node = end_node
        
        for i in trange(self.max_iter, disable=not pbar):
            rand_node = self.get_rand_node()
            near_node, _ = self.get_nearest_neighbor(rand_node)
            new_node = self.get_new_node(near_node, rand_node)
            
            if col_checker(new_node.p):
                continue
            else:
                self.nodes.append(new_node)
                near_node.connect(new_node)
                
                min_node, min_dist = self.get_nearest_neighbor(self.goal_node)
                if min_dist < self.step_size:
                    self.nodes.append(self.goal_node)
                    min_node.connect(self.goal_node)
                    self.planned = True
                    break
        
        if self.planned:
            tmp_node = self.goal_node
            while True:    
                if tmp_node is None:
                    break
                tmp_node.onpath = True
                tmp_node = tmp_node.parent
        else:
            print('Planning Failed.')
                
        gc.enable()

    def plot(self):
        assert self.dim == 2, f'plotting is only for 2 dim. space, now {self.dim}.'
        
        points = torch.cat([x.p.view(1, self.dim) for x in self.nodes], dim=0).detach().cpu()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', showlegend=False))
        for node in self.nodes:
            if node.parent is not None:
                if node.onpath:
                    line_color = 'black'
                else:
                    line_color = 'red'
                fig.add_trace(go.Scatter(x=[node.p[0].detach().cpu(), node.parent.p[0].detach().cpu()], 
                                         y=[node.p[1].detach().cpu(), node.parent.p[1].detach().cpu()], 
                                         mode='lines', line_color=line_color, showlegend=False))
        
        fig.update_layout(**plotly_layout, width=400, height=400)
        return fig
    
    def get_path(self):
        points = []
        tmp_node = self.goal_node
        while True:    
            if tmp_node is None:
                break
            points.append(tmp_node.p.view(1, self.dim))
            tmp_node = tmp_node.parent
        points = torch.cat(points[::-1], dim=0)
        return points

    def get_nearest_neighbor(self, node):
        
        dist = self.dist_batch(self.nodes, [node])
        min_dist = dist.min(dim=0).values
        nearest_node = self.nodes[dist.min(dim=0).indices.item()]
                
        return nearest_node, min_dist
    
    def get_rand_node(self):
        if torch.rand(1) > self.goal_sample_rate:
            rand_p = torch.rand(self.dim).to(self.device) * (self.q_max - self.q_min) + self.q_min
            return Node(rand_p).to(self.device)
        else:
            return self.goal_node
    
    def get_new_node(self, nearest_node, ran_node):
        distance = self.dist(nearest_node, ran_node)
        ratio = self.step_size / distance
        new_p = (1-ratio)*nearest_node.p + ratio*ran_node.p
        return Node(new_p)
    
    @staticmethod
    def dist(node1, node2):
        return torch.norm(node1.p-node2.p)
    
    @staticmethod
    def dist_batch(nodes1, nodes2):
        p1s = torch.cat([n.p.view(1, -1) for n in nodes1], dim=0)
        p2s = torch.cat([n.p.view(1, -1) for n in nodes2], dim=0)

        dim = p1s.shape[-1]

        N = len(p1s)
        M = len(p2s)

        p1s = p1s.unsqueeze(1).repeat(1, M, 1).reshape(-1, dim)
        p2s = p2s.unsqueeze(0).repeat(N, 1, 1).reshape(-1, dim)

        dist = p1s - p2s
        # dist[dist<0] = dist[dist<0] + 2*torch.pi
        # dist[dist>torch.pi] = 2*torch.pi - dist[dist>torch.pi]  
        dist = torch.norm(dist, dim=1).reshape(N, M)
        return dist
    
class RRTConnectplanar:
    def __init__(self, step_size, q_min, q_max, device, goal_sample_rate=0.05, max_iter=10000):
        self.step_size = step_size
        self.q_min = torch.as_tensor(q_min).to(device)
        self.q_max = torch.as_tensor(q_max).to(device)
        self.dim = len(q_min)
        self.nodes1 = []
        self.nodes2 = []
        
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.planned = False
        self.device = device
    
    def planning(self, start, end, col_checker, pbar=False, verbose=True):
        gc.disable()
        assert len(start) == len(end) == self.dim, f'expected dim. of input is {self.dim}, got {len(start)}'
        self.planned = False
        
        start_node = Node(start).to(self.device)
        end_node = Node(end).to(self.device)
        self.nodes1.append(start_node)
        self.nodes2.append(end_node)
        
        self.goal_node = end_node
        
        self.nodes_reversed = False
        
        if verbose:
            print(now() + ' Planning Start.')
            
        i_colcheck = 0
        
        for _ in trange(self.max_iter, disable=not pbar, desc='RRTing', ncols=100):
            rand_node = self.get_rand_node()
            near_node, _ = self.get_nearest_neighbor(self.nodes1, rand_node)
            new_node = self.get_new_node(near_node, rand_node)
            
            i_colcheck += 1
            if not col_checker(new_node.p):
                self.nodes1.append(new_node)
                near_node.connect(new_node)
                
                node_near_prim, _ = self.get_nearest_neighbor(self.nodes2, new_node)
                node_new_prim = self.get_new_node(node_near_prim, new_node)
                
                i_colcheck += 1
                if not col_checker(node_new_prim.p):
                    self.nodes2.append(node_new_prim)
                    node_near_prim.connect(node_new_prim)
                    
                    for _ in range(1000):
                        node_new_prim2 = self.get_new_node(node_new_prim, new_node)
                        
                        i_colcheck += 1
                        if not col_checker(node_new_prim2.p):
                            self.nodes2.append(node_new_prim2)
                            node_new_prim.connect(node_new_prim2)
                            node_new_prim = node_new_prim2
                        else:
                            break
                            
                        if self.dist(node_new_prim, new_node) < self.step_size:
                            break

                    min_dist = self.dist(node_new_prim, new_node)
                    if min_dist < self.step_size:
                        self.planned = True
                        break

            if len(self.nodes2) < len(self.nodes1):
                list_tmp = self.nodes2
                self.nodes2 = self.nodes1
                self.nodes1 = list_tmp
                self.nodes_reversed = not self.nodes_reversed
        
        if self.planned:
            if verbose:
                print(now() + f' Planning End: {i_colcheck} times of collision checking.')
            tmp_node1, tmp_node2 = new_node, node_new_prim
            while True:    
                if tmp_node1 is None:
                    break
                tmp_node1.onpath = True
                tmp_node1 = tmp_node1.parent
                
            while True:    
                if tmp_node2 is None:
                    break
                tmp_node2.onpath = True
                tmp_node2 = tmp_node2.parent
        else:
            if verbose:
                print(now() + f' Planning Failed.')
            return False
            
        self.end_node1 = new_node
        self.end_node2 = node_new_prim
                
        gc.enable()
        
        return i_colcheck

    def plot(self):
        assert self.dim == 2, f'plotting is only for 2 dim. space, now {self.dim}.'
        
        points = torch.cat([x.p.view(1, self.dim) for x in self.nodes1+self.nodes2], dim=0).detach().cpu()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points[:len(self.nodes1), 0], y=points[:len(self.nodes1), 1], mode='markers', marker_color='blue', showlegend=False))
        fig.add_trace(go.Scatter(x=points[len(self.nodes1):, 0], y=points[len(self.nodes1):, 1], mode='markers', marker_color='red', showlegend=False))
        for node in self.nodes1 + self.nodes2:
            if node.parent is not None:
                if node.onpath:
                    line_color = 'black'
                else:
                    line_color = 'white'
                fig.add_trace(go.Scatter(x=[node.p[0].detach().cpu(), node.parent.p[0].detach().cpu()], 
                                         y=[node.p[1].detach().cpu(), node.parent.p[1].detach().cpu()], 
                                         mode='lines', line_color=line_color, showlegend=False))
        
        fig.update_layout(**plotly_layout, width=400, height=400)
        return fig
    
    def get_path(self):
        points1 = []
        tmp_node = self.end_node1
        while True:    
            if tmp_node is None:
                break
            points1.append(tmp_node.p.view(1, self.dim))
            tmp_node = tmp_node.parent
            
        points2 = []
        tmp_node = self.end_node2
        while True:    
            if tmp_node is None:
                break
            points2.append(tmp_node.p.view(1, self.dim))
            tmp_node = tmp_node.parent
            
        if self.nodes_reversed:
            points = torch.cat(points2[::-1]+points1, dim=0)
        else:
            points = torch.cat(points1[::-1]+points2, dim=0)
        return points

    def get_nearest_neighbor(self, nodes, node):

        dist = self.dist_batch(nodes, [node])
        min_dist = dist.min(dim=0).values
        nearest_node = nodes[dist.min(dim=0).indices.item()]
                
        return nearest_node, min_dist
    
    def get_rand_node(self):
        rand_p = torch.rand(self.dim).to(self.device) * (self.q_max - self.q_min) + self.q_min
        return Node(rand_p).to(self.device)
    
    def get_new_node(self, nearest_node, ran_node):
        distance = self.dist(nearest_node, ran_node)
        ratio = self.step_size / distance
        new_p = (1-ratio)*nearest_node.p + ratio*ran_node.p
        return Node(new_p)
    
    @staticmethod
    def dist(node1, node2):
        return torch.norm(node1.p-node2.p)
    
    @staticmethod
    def dist_batch(nodes1, nodes2):
        p1s = torch.cat([n.p.view(1, -1) for n in nodes1], dim=0)
        p2s = torch.cat([n.p.view(1, -1) for n in nodes2], dim=0)

        dim = p1s.shape[-1]

        N = len(p1s)
        M = len(p2s)

        p1s = p1s.unsqueeze(1).repeat(1, M, 1).reshape(-1, dim)
        p2s = p2s.unsqueeze(0).repeat(N, 1, 1).reshape(-1, dim)

        dist = p1s - p2s
        # dist[dist<0] = dist[dist<0] + 2*torch.pi
        # dist[dist>torch.pi] = 2*torch.pi - dist[dist>torch.pi]  
        dist = torch.norm(dist, dim=1).reshape(N, M)
        return dist
    
    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node(node_new_prim2.p)
        node_new.parent = node_new_prim
        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.p == node_new.p:
            return True
        return False
    
if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    planner = RRTConnectplanar(step_size=0.05, q_min=[0]*2, q_max=[1]*2, device=device)
    
    def col_checker(x):
        return torch.norm(x - 0.8) < 0.1 or torch.norm(x - 0.2) < 0.1
    
    planner.planning([0]*2, [1]*2, col_checker, pbar=True)

    fig = planner.plot()
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=0.7, y0=0.7, x1=0.9, y1=0.9,
    )
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=0.1, y0=0.1, x1=0.3, y1=0.3,
    )
    fig.show()
