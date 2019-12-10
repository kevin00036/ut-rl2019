import numpy as np
import torch
import time

class Planner:
    def __init__(self, trans_fn, gamma, use_td3=True, device='cpu'):
        self.device = device
        self.trans_fn = trans_fn # (s, s') -> (G(s, s'), R(s, s'), Pi(s, s'))
        self.gamma = gamma
        self.use_td3 = use_td3

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
        self.Gt, self.Rt, self.Pit = self.trans_fn(self.waypoints, self.waypoints, target=True)
        self.Gt = self.Gt.clamp(0., self.gamma)

        # mask = (self.G >= 0.8).float()
        # self.G = self.G * mask
        # self.R = self.R + (1-mask) * (-1E10)
        
        print(f'Avg R: {float(self.R.mean()):.5f}, Avg G: {float(self.G.mean()):.5f}')

    def pre_plan(self):
        self.V = torch.zeros(self.n, device=self.device)

        thres = 1e-5

        it = 0
        plan_steps = 20
        with torch.no_grad():
            while True:
                it += 1

                Vnt = self.Rt + self.Gt * self.V.unsqueeze(0)
                Vn = self.R + self.G * self.V.unsqueeze(0)
                maxsp = Vn.max(dim=1)[1]
                if self.use_td3:
                    Vm = torch.min(self.R, self.Rt) + torch.min(self.G, self.Gt) * self.V.unsqueeze(0)
                else:
                    Vm = self.R + self.G * self.V.unsqueeze(0)
                newV = Vm.gather(1, maxsp.unsqueeze(1)).squeeze(1)

                diff = (newV - self.V).abs().max()
                self.V = newV

                if diff < thres or it >= plan_steps:
                    break
                
                if it % 200 == 0:
                    print(f'{it} Iterations, Vmean = {float(self.V.mean()):.5f}')


        print(f'{it} Iterations, Vmean = {float(self.V.mean()):.5f}')


    def plan(self, s, get_state=False):
        s = torch.from_numpy(s).float().to(self.device).unsqueeze(0)
        G, R, Pi = self.trans_fn(s, self.waypoints)
        Gt, Rt, Pit = self.trans_fn(s, self.waypoints, target=True)
        Vs = (R + G * self.V.unsqueeze(0)).squeeze(0)
        if self.use_td3:
            Vm = (torch.min(R, Rt) + torch.min(G, Gt) * self.V.unsqueeze(0)).squeeze(0)
        else:
            Vm = (R + G * self.V.unsqueeze(0)).squeeze(0)
        
        g_idx = Vs.argmax()
        a = Pi[0, g_idx]
        a = int(a)
        v = float(Vm[g_idx])
        # print('Act', a, 'PredRew', float(Vs[g_idx]))
        if get_state:
            return self.waypoints[g_idx].cpu().numpy()
        else:
            return a, v

    def show_plan(self, s, env, step=20):

        s_list = [s]
        for i in range(step):
            s = self.plan(s, get_state=True)
            s_list.append(s)

        s_list = np.array(s_list)
        # print(s_list)

        for st in s_list:
            env.env.state = st
            env.render()
            time.sleep(0.05)




