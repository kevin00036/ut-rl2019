import numpy as np
import torch

class Planner:
    def __init__(self, trans_fn, gamma, device='cpu'):
        self.device = device
        self.trans_fn = trans_fn # (s, s') -> (G(s, s'), R(s, s'), Pi(s, s'))
        self.gamma = gamma
        self.waypoints = None
        self.n = None
        self.R = None
        self.G = None
        self.Pi = None
        self.V = None

    def set_waypoint_states(self, waypoints):
        self.waypoints = torch.from_numpy(waypoints).float().to(self.device) # Npts * dim
        self.n = self.waypoints.shape[0]

    def update_trans(self):
        self.G, self.R, self.Pi = self.trans_fn(self.waypoints, self.waypoints)
        self.G = self.G.clamp(0., self.gamma)
        
        print(f'Avg R: {float(self.R.mean()):.5f}, Avg G: {float(self.G.mean()):.5f}')

    def pre_plan(self):
        self.V = torch.zeros(self.n, device=self.device)

        thres = 1e-5

        it = 0
        with torch.no_grad():
            while True:
                it += 1

                Vn = self.R + self.G * self.V.unsqueeze(0)
                newV = Vn.max(dim=1)[0]
                # newV = Vn.topk(k=10, dim=1)[0].min(dim=1)[0]

                diff = (newV - self.V).abs().max()
                self.V = newV

                if diff < thres:
                    break
                
                if it % 200 == 0:
                    print(f'{it} Iterations, Vmean = {float(self.V.mean()):.5f}')


        print(f'{it} Iterations, Vmean = {float(self.V.mean()):.5f}')


    def plan(self, s):
        s = torch.from_numpy(s).float().to(self.device).unsqueeze(0)
        G, R, Pi = self.trans_fn(s, self.waypoints)
        Vs = (R + G * self.V.unsqueeze(0)).squeeze(0)
        
        g_idx = Vs.argmax()
        a = Pi[0, g_idx]
        a = int(a)
        # print('Act', a, 'PredRew', float(Vs[g_idx]))
        return a



