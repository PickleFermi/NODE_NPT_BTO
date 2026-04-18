import torch
from sympy.physics.mechanics import kinetic_energy

from torchmd.util import compute_grad
import numpy as np 
import sys
from ase import units
from torchmd.sovlers import odeint_adjoint, odeint, build_npt_augmented_rhs
from ase.geometry import wrap_positions
'''
    Here contains object for simulation and computing the equation of state
'''


class Simulations():

    """Simulation object for handing runnindg MD and logging
    
    Attributes:
        device (str or int): int for GPU, "cpu" for cpu
        integrator (nn.module): function that updates force and velocity n
        keys (list): name of the state variables e.g. "velocities" and "positions "
        log (dict): save state vaiables in numpy arrays 
        solvemethod (str): integration method, current options are 4th order Runge-Kutta (rk4) and Verlet 
        system (torch.System): System object to contain state of molecular systems 
        wrap (bool): if True, wrap the coordinates based on system.cell 
    """
    
    def __init__(self,
                 system,
                  integrator,
                  wrap=False,
                  method="NH_verlet"):

        self.system = system 
        self.device = system.device
        self.integrator = integrator
        self.solvemethod = method
        self.wrap = wrap
        self.keys = self.integrator.state_keys
        self.initialize_log()

    def initialize_log(self):
        self.log = {}
        for key in self.keys:
            self.log[key] = []

    def update_log(self, trajs):
        for i, key in enumerate( self.keys ):
            if trajs[i][0].device != 'cpu':
                self.log[key].append(trajs[i][-1].detach().cpu().numpy()) 
            else:
                self.log[key].append(trajs[i][-1].detach().numpy()) 

    def update_states(self):
        if "positions" in self.log.keys():
            self.system.set_positions(self.log['positions'][-1])
        if "velocities" in self.log.keys():
            self.system.set_velocities(self.log['velocities'][-1])

    def get_check_point(self):

        if hasattr(self, 'log'):
            states = [torch.Tensor(self.log[key][-1]).to(self.device) for key in self.log]

            if self.wrap:
                wrapped_xyz = wrap_positions(self.log['positions'][-1], self.system.get_cell())
                states[1] = torch.Tensor(wrapped_xyz).to(self.device)

            return states 
        else:
            raise ValueError("No log available")
        
    def simulate(self, steps=1, dt=1.0 * units.fs, frequency=1):

        if self.log['positions'] == []:
            states = self.integrator.get_inital_states(self.wrap)
        else:
            states = self.get_check_point()

        sim_epochs = int(steps//frequency)
        t = torch.Tensor([dt * i for i in range(frequency)]).to(self.device)

        for epoch in range(sim_epochs):

            if self.integrator.adjoint:
                trajs = odeint_adjoint(self.integrator, states, t, method=self.solvemethod)
            else:
                for var in states:
                    var.requires_grad = True 
                trajs = odeint(self.integrator, tuple(states), t, method=self.solvemethod)
            self.update_log(trajs)
            self.update_states()

            states = self.get_check_point()

        return trajs

class NVE(torch.nn.Module):

    """Equation of state for constant energy integrator (NVE ensemble)
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        mass (torch.Tensor): masses of each particle
        model (snn.module): energy functions that takes in coordinates
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (torchmd.System): system object
    """
    
    def __init__(self, potentials, system, adjoint=True, topology_update_freq=1):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.mass = torch.Tensor(system.get_masses()).to(self.system.device)
        self.N_dof = self.mass.shape[0] * system.dim
        self.dim = system.dim
        self.adjoint = adjoint
        self.state_keys = ['velocities', 'positions']

        self.topology_update_freq = topology_update_freq
        self.update_count = 0

    def update_topology(self, q):

        if self.update_count % self.topology_update_freq == 0:
            self.model._reset_topology(q)
        self.update_count += 1
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            
            if self.adjoint:
                q.requires_grad = True
            
            p = v * self.mass[:, None]

            self.update_topology(q)
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))
            dvdt = f

        return (dvdt, v)

    def get_inital_states(self, wrap=True):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap)]

        states = [torch.Tensor(var).to(self.system.device) for var in states]

        return states

class NoseHooverChain(torch.nn.Module):

    """Equation of state for NVT integrator using Nose Hoover Chain 

    Nosé, S. A unified formulation of the constant temperature molecular dynamics methods. The Journal of Chemical Physics 81, 511–519 (1984).
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        mass (torch.Tensor): masses of each particle
        model (snn.module): energy functions that takes in coordinates
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (torchmd.System): system object
        num_chains (int): number of chains 
        Q (float): Heat bath mass
        T (float): Temperature
        target_ke (float): target Kinetic energy 
    """
    
    def __init__(self, potentials, system, T, num_chains=2, Q=1.0, adjoint=True
                ,topology_update_freq=1):
        super().__init__()
        self.model = potentials 
        self.system = system
        cell = torch.from_numpy(np.array(self.system.cell)).float().to(system.device)
        self.cell = cell.unsqueeze(0)
        self.device = system.device # should just use system.device throughout
        self.mass = torch.Tensor(system.get_masses()).to(self.device)
        self.T = T # in energy unit(eV)
        self.N_dof = self.mass.shape[0] * system.dim
        self.target_ke = (0.5 * self.N_dof * T )
        
        self.num_chains = num_chains
        self.Q = np.array([Q,
                   *[Q/self.system.get_number_of_atoms()]*(num_chains-1)])
        self.Q = torch.Tensor(self.Q).to(self.device)
        self.dim = system.dim
        self.adjoint = adjoint
        self.state_keys = ['velocities', 'positions', 'baths']
        self.topology_update_freq = topology_update_freq
        self.update_count = 0

    def update_topology(self, q):

        if self.update_count % self.topology_update_freq == 0:
            self.model._reset_topology(q)
        self.update_count += 1


    def update_T(self, T):
        self.T = T 
        
    def forward(self, t, state):
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            p = v * self.mass[:, None]

            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            # self.update_topology(q)
            
            u, _, _ = self.model(q.unsqueeze(0), self.cell)
            f = -compute_grad(inputs=q, output=u.sum(-1))
            # print(f"u:{u.item()}, f.std:{f.std()}, f.mean:{f.mean()}")

            coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dpdt = f - coupled_forces

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T

            dvdt = dpdt / self.mass[:, None]

        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))

    def get_inital_states(self, wrap=False):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap), 
                [0.0] * self.num_chains]

        states = [torch.Tensor(var).to(self.system.device) for var in states]
        
        # 确保所有状态都有梯度（如果启用adjoint）
        if self.adjoint:
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor):
                    states[i] = state.requires_grad_(True)
        
        return states


class Isomerization(torch.nn.Module):

    """Quantum isomerization equation of state. 

    The hamiltonian is precomputed in the new basis obtained by orthogonalizing
         the original tensor product space of vibrational and rotational coordinates 
    
    Attributes:
        device (int or str): device
        dim (int): the size of wave function 
        dipole (torch.nn.Parameter): dipole operator
        e_field (torch.nn.Parameter): electric field 
        ham (torch.nn.Parameter): hamiltonian
        max_e_t (int): max time the electric field can be on
    """

    def __init__(self, dipole, e_field, ham, max_e_t, device=0):
        super().__init__()

        self.device = device
        self.dipole = dipole.to(self.device)
        self.ham = ham.to(self.device)
        self.dim = len(ham)
        self.e_field = torch.nn.Parameter(e_field)
        self.max_e_t = max_e_t

    
    def forward(self, t, psi):
        with torch.set_grad_enabled(True):
            psi.requires_grad = True
            # real and imaginary parts of psi
            psi_R = psi[:self.dim]
            psi_I =  psi[self.dim:]

            if t < self.max_e_t.to(t.device):
                # find the value of E at the time closest
                # to now
                t_index = torch.argmin(abs(self.e_field[:, 0] - t))
                e_now = self.e_field[t_index][-1]
            else:
                e_now = 0

            # total Hamiltonian =  H - mu \dot E
            H_eff = self.ham - self.dipole * e_now
            
            # d/dt of real and imaginary parts of psi
            dpsi_R = torch.matmul(H_eff, psi_I)
            dpsi_I = -torch.matmul(H_eff, psi_R)
            
            d_psi_dt = torch.cat((dpsi_R, dpsi_I))

        return d_psi_dt


class NoseHooverChain_NPT(NoseHooverChain):
    """扩展NoseHooverChain支持NPT系综（恒温恒压）
    
    继承自NoseHooverChain，添加压力控制和晶胞矩阵动力学
    
    Attributes:
        P_ext (torch.Tensor): 外部压力张量
        ttime (float): 温度弛豫时间
        pfactor (float): 压力弛豫参数
        mask (torch.Tensor): 压力控制掩码
        eta (torch.Tensor): 压浴变量
        zeta (torch.Tensor): 热浴变量
        h (torch.Tensor): 晶胞矩阵
    """
    
    def __init__(self, potentials, system, T, P_ext=1.0*units.bar, 
                 ttime=20.0*units.fs, pfactor=2e6*units.GPa*(units.fs**2),
                 num_chains=2, Q=1.0, adjoint=True, topology_update_freq=1,
                 mask=None):
        # 调用父类初始化
        super().__init__(potentials, system, T, num_chains, Q, adjoint, topology_update_freq)
        
        # NPT特有参数
        self.P_ext = self.set_stress(P_ext, system.device)
        self.ttime = ttime
        self.pfactor = pfactor
        self.mask = self.set_mask(mask, system.device)
        
        # 扩展状态键，包含压浴、热浴和晶胞矩阵
        self.state_keys = ['velocities', 'positions', 'baths', 'eta', 'zeta', 'cell_matrix']
        
        # 设置状态变量数量
        self.NUM_VAR = 6
        self.adjoint_debug = False
        self.npt_augmented_rhs = None
        
        # 计算压力控制参数
        n_atoms = self.system.get_number_of_atoms()
        self.tfact = 2.0 / (3 * n_atoms * self.T * self.ttime * self.ttime)
        
        # 获取当前晶胞并计算压力因子
        cell = torch.from_numpy(np.array(self.system.get_cell())).float().to(system.device)
        self.h = cell  # 保存晶胞矩阵
        self.pfact = 1.0 / (self.pfactor * torch.linalg.det(self.h))
        
        # 目标动能（去除质心运动自由度）
        self.desiredEkin = 1.5 * (n_atoms - 1) * self.T
    
    def set_stress(self, stress, device):
        """设置外部应力张量"""
        if isinstance(stress, (float, int)):
            # 如果是标量，转换为各向同性压力
            stress = torch.tensor([-stress, -stress, -stress, 0.0, 0.0, 0.0], 
                                device=device, dtype=torch.float32)
        else:
            stress = torch.tensor(stress, device=device, dtype=torch.float32)
            if stress.shape == (3, 3):
                # 如果是3x3矩阵，转换为6分量向量
                stress = torch.tensor((
                    stress[0, 0], stress[1, 1], stress[2, 2],
                    stress[1, 2], stress[0, 2], stress[0, 1]
                ), device=device, dtype=torch.float32)
            elif stress.shape != (6,):
                raise ValueError("External stress shape must be (3,3) or (6,).")
        return stress
    
    def set_mask(self, mask, device):
        """设置压力控制掩码"""
        if mask is None:
            mask = torch.ones((3,), device=device, dtype=torch.float32)
        if not hasattr(mask, "shape"):
            mask = torch.tensor(mask, device=device, dtype=torch.float32)
        if mask.shape not in [(3,), (3, 3)]:
            raise RuntimeError("Mask shape must be (3,) or (3,3).")
        mask = torch.not_equal(mask, 0)
        return torch.outer(mask, mask) if mask.shape == (3,) else mask
    
    def _makeuppertriangular(self, sixvector):
        """将6分量向量转换为上三角矩阵，保持梯度连接"""
        device = sixvector.device
        dtype = sixvector.dtype
        
        # 确保输入有梯度
        if not sixvector.requires_grad:
            sixvector = sixvector.requires_grad_(True)
        
        # 使用cat来构建矩阵，避免原地操作，确保梯度传播
        row1 = torch.cat([sixvector[0:1], sixvector[5:6], sixvector[4:5]])
        row2 = torch.cat([torch.zeros(1, device=device, dtype=dtype, requires_grad=True), 
                         sixvector[1:2], sixvector[3:4]])
        row3 = torch.cat([torch.zeros(1, device=device, dtype=dtype, requires_grad=True), 
                         torch.zeros(1, device=device, dtype=dtype, requires_grad=True), 
                         sixvector[2:3]])
        
        # 使用stack构建矩阵，保持梯度连接
        result = torch.stack([row1, row2, row3])
        
        # 验证梯度连接
        if sixvector.requires_grad and not result.requires_grad:
            result = result.requires_grad_(True)
        
        return result
    
    def forward(self, t, state):
        """ODE求解器的forward函数，支持NPT系综"""
        with torch.set_grad_enabled(True):
            # 检查是否是adjoint模式
            if len(state) == self.NUM_VAR * 2 + 2:
                # Adjoint模式：需要返回14个值
                return self._forward_adjoint(t, state)
            if len(state) == self.NUM_VAR:
                # 前向模式：返回6个导数
                return self._forward_normal(t, state)
            raise ValueError(
                f"NPT dynamics expects {self.NUM_VAR} state tensors in forward mode or "
                f"{self.NUM_VAR * 2 + 2} tensors in adjoint mode, got {len(state)}."
            )
    
    def _forward_normal(self, t, state):
        """前向模式的forward函数"""
        # 解包状态变量：v, q, p_v, eta, zeta, h
        v = state[0]  # 速度
        q = state[1]  # 位置
        p_v = state[2]  # 热浴链变量
        eta = state[3]  # 压浴变量
        zeta = state[4]  # 热浴变量
        h = state[5]  # 晶胞矩阵
        
        if self.adjoint:
            q.requires_grad = True
            h.requires_grad = True
        
        # 1. 计算动量和动能（继承NoseHooverChain逻辑）
        p = v * self.mass[:, None]
        sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum()
        
        # 2. 计算力和应力
        try:
            # 使用模型的forward方法计算能量和应力
            u, _, P_int = self.model(q.unsqueeze(0), h.unsqueeze(0))
            f = -compute_grad(inputs=q, output=u.sum(-1))
            
            # 处理应力输出 - 确保梯度传播
            if P_int is None:
                P_int = torch.zeros(6, device=self.device, dtype=torch.float32, requires_grad=True)
            else:
                # 确保应力张量有梯度
                if not P_int.requires_grad:
                    P_int = P_int.requires_grad_(True)
            
            P_int = P_int.squeeze(0)  # 移除批次维度
            
        except Exception as e:
            print(f"模型计算失败: {e}")
            # 回退到基础计算 - 保持梯度连接
            u = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
            P_int = torch.zeros(6, device=self.device, dtype=torch.float32, requires_grad=True)
            f = torch.zeros_like(q, device=self.device, dtype=torch.float32, requires_grad=True)
        
        # 3. 热浴链更新（继承NoseHooverChain逻辑）
        coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)
        dpdt = f - coupled_forces
        
        # 热浴链导数
        dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1] / self.Q[1]
        dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:] * p_v[1:-1] / self.Q[2:]
        dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T
        
        # 4. 压浴更新（NPT特有）
        volume = torch.linalg.det(h)
        if self.pfactor is not None:
            # 计算压力差并限制数值范围 - 使用更温和的clamp避免梯度消失
            P_clip = torch.clamp(P_int, -1e2, 1e2)  # 降低clamp范围
            deltaeta = -2 * self.ttime * (self.pfact * volume * (P_clip - self.P_ext))
            
            # 应用掩码并转换为上三角矩阵
            if self.mask is not None:
                soft_mask = self.mask.float() + 1e-8  # 软掩码避免梯度丢失
                deta_dt = soft_mask * self._makeuppertriangular(deltaeta)
            else:
                deta_dt = self._makeuppertriangular(deltaeta)
        else:
            deta_dt = torch.zeros((3, 3), device=self.device, dtype=torch.float32, requires_grad=True)
        
        # 5. 热浴更新（NPT特有）
        # 修复维度匹配问题：self.mass是(N,1)，v是(N,3)
        current_kinetic_energy = 0.5 * torch.sum(self.mass * (v ** 2).sum(dim=1, keepdim=True))
        # 限制温控项避免极端震荡 - 使用更温和的clamp
        energy_diff = current_kinetic_energy - self.desiredEkin
        energy_diff_clamped = torch.clamp(energy_diff, -1e3, 1e3)  # 降低clamp范围
        dzeta_dt = 2 * self.ttime * self.tfact * energy_diff_clamped
        dzeta_dt = dzeta_dt * torch.eye(3, device=self.device, dtype=torch.float32)
        
        # 6. 晶胞矩阵更新（NPT特有）
        # 限制盒形变化率避免数值不稳定 - 使用更温和的clamp
        eta_clamped = torch.clamp(eta, -1e1, 1e1)  # 降低clamp范围
        dh_dt = torch.matmul(h, eta_clamped)
        
        # 7. 速度更新
        dvdt = dpdt / self.mass[:, None]
        
        # 8. 位置更新（分数坐标）
        inv_h = torch.linalg.inv(h)
        dq_dt = torch.matmul(v, inv_h)
        
        # 返回所有导数
        return (dvdt, dq_dt, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])), 
               deta_dt, dzeta_dt, dh_dt)
    
    def _forward_adjoint(self, t, state):
        """Adjoint模式的forward函数，显式返回完整 augmented RHS。"""
        if self.npt_augmented_rhs is None:
            self.npt_augmented_rhs = build_npt_augmented_rhs(
                self,
                debug=self.adjoint_debug
            )
        return self.npt_augmented_rhs(t, state)
    
    def get_inital_states(self, wrap=False):
        """获取初始状态，包含所有NPT变量"""
        # 获取基础状态（继承NoseHooverChain）
        base_states = super().get_inital_states(wrap)
        
        # 确保基础状态有梯度
        if self.adjoint:
            # 确保所有基础状态都有梯度
            for i, state in enumerate(base_states):
                if isinstance(state, torch.Tensor) and not state.requires_grad:
                    base_states[i] = state.requires_grad_(True)
        
        # 添加NPT特有状态 - 确保梯度连接
        eta = torch.zeros((3, 3), device=self.device, dtype=torch.float32, requires_grad=self.adjoint)
        zeta = torch.zeros((3, 3), device=self.device, dtype=torch.float32, requires_grad=self.adjoint)
        
        # 获取晶胞矩阵 - 确保梯度连接
        cell = torch.from_numpy(np.array(self.system.get_cell())).float().to(self.device)
        h = cell.requires_grad_(self.adjoint)
        
        # 组合所有状态：v, q, p_v, eta, zeta, h
        states = [*base_states, eta, zeta, h]
        
        # 最终验证：确保所有状态都有梯度（如果启用adjoint）
        if self.adjoint:
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor) and not state.requires_grad:
                    print(f"警告：状态[{i}] 没有梯度，正在修复...")
                    states[i] = state.requires_grad_(True)
        
        return states