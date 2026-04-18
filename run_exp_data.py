import argparse
import random
from datetime import datetime
from ase.io import read
import numpy as np
import torch
import os
from ase import Atoms, units
from ase.build import make_supercell
from matplotlib import pyplot as plt
from scipy import interpolate
from model.dimenet.dimenet_pbc import DimeNet
from model.p0_data_idx import idx_calculator
from torchmd.interface import DimeNetPotentials, GNNPotentials, Stack
from torchmd.md import NoseHooverChain, Simulations, NoseHooverChain_NPT
from torchmd.observable import pdf
from torchmd.sovlers import build_npt_augmented_rhs
from torchmd.system import System
import matplotlib
matplotlib.use('Agg')


# 1.NPT   2.EXP & DFT fused
parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str, default='result')
parser.add_argument("-model_type", type=str, default='DimeNet')
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-size", type=int, default=4)
parser.add_argument("-cutoff", type=int, default=4)
parser.add_argument("-nbins", type=int, default=64)
parser.add_argument("-lr", type=int, default=1e-6)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-opt_freq", type=int, default=150)
params = vars(parser.parse_args())


def set_seed(seed=40):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

# set_seed()
if torch.cuda.is_available():
    device = params['device']
else:
    device = 'cpu'

size = params['size']
model_type = params['model_type']
cutoff = params['cutoff']
nbins = params['nbins']
epochs = params['epochs']
opt_freq = params['opt_freq']
ADJOINT_DEBUG = False

now = datetime.now()
dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))
model_path = '{}/{}'.format(params['logdir'], model_type + dt_string)
os.makedirs(model_path)

def build_structure(n_cells):
    a0 = 8.06187 / 2
    n1, n2, n3 = n_cells[0], n_cells[1], n_cells[2]
    # build atoms for perovskite.
    pos = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0]])
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # self.cell = np.diag(np.array(n_cells)*self.a0)

    pri_atoms = Atoms(symbols=['Ba', 'Ti', 'O', 'O', 'O'], positions=a0 * pos,
                      cell=a0 * cell, pbc=True)
    atoms = make_supercell(prim=pri_atoms, P=[[n1, 0, 0], [0, n2, 0], [0, 0, n3]])
    return atoms


def get_system(T, size):
    atoms = build_structure((size, size, size))
    system = System(atoms, device=device)
    system.set_temperature(T * units.kB)
    return system


def get_observer(system, T, nbins):

    data_path = "D:\\paper_by_myself\\NPT\\BTO\\BaTiO3_data\\" + str(T) + "k.txt"
    data = np.loadtxt(data_path)[:, :-1]
    start = 0.02
    end = 8.0
    data = data[(data[:, 0] >= start) & (data[:, 0] <= end)]
    bins = data[:, 0]
    nbins = len(bins) - 1
    obs = pdf(system, nbins, bins, (start, end))
    f = interpolate.interp1d(data[:, 0], data[:, 1])
    g_obs = torch.Tensor(f(bins[:-1])).to(device)
    return bins, g_obs, obs


def compute_D(dev, rho, rrange):
    if not isinstance(dev, torch.Tensor):
        dev = torch.tensor(dev)
    if not isinstance(rho, torch.Tensor):
        rho = torch.tensor(rho)
    if not isinstance(rrange, torch.Tensor):
        rrange = torch.tensor(rrange)
    pi = torch.tensor(np.pi, device=dev.device)
    return (4 * pi * rho * (rrange ** 2) * dev ** 2 * (rrange[2] - rrange[1])).sum()


def JS_rdf(g_obs, g):
    e0 = torch.tensor(1e-4, device=g_obs.device)
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js


def plot_rdfs(bins, target_g, simulated_g, fname, path, pname=None, save=False, loss=None):
    plt.title("epoch {}".format(pname))
    bins_np = bins[:-1].detach().cpu().numpy()
    sim_g_np = simulated_g.detach().cpu().numpy()
    target_g_np = target_g.detach().cpu().numpy()
    
    plt.plot(bins_np, sim_g_np, linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins_np, target_g_np, linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel(r"$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + f'/{fname}' + f'_{loss.item()}' + '.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()

    if save:
        data = np.vstack((bins_np, sim_g_np))
        np.savetxt(path + '/{}.csv'.format(fname), data, delimiter=',')


T = 250
system = get_system(T, size)

loader = idx_calculator(cutoff=0.9, n_cells=(size,size,size), atoms=system)
idx_dict_cpu = loader.idx_dict
idx_dict = {k: v.to(device) for k, v in idx_dict_cpu.items()}
prior = DimeNet(idx_dict=idx_dict, meV2eV=True).to(device)  # , meV2eV=True
if device == 'cpu':
    prior.load_state_dict(torch.load('model/net_prms_best.pkl', map_location=torch.device('cpu')))
else:
    prior.load_state_dict(torch.load('model/net_prms_best.pkl'))
prior.eval()
net = DimeNet(idx_dict=idx_dict, meV2eV=True).to(device)

pair = DimeNetPotentials(system, prior).to(device)
NN = GNNPotentials(system, net, cutoff=cutoff, model_type=model_type)
model = Stack({'nn': NN, 'pair': pair}, atoms=system)

solver_method = 'NH_verlet_NPT'
if solver_method == "NH_verlet":
    integrator = NoseHooverChain(model,
                                system,
                                Q=50.0,
                                T=T * units.kB,
                                num_chains=5,
                                adjoint=True,  
                                topology_update_freq=1).to(system.device)
else:
    integrator = NoseHooverChain_NPT(
        model,
        system,
        T=T * units.kB,
        P_ext=1.0 * units.bar,
        ttime=50.0 * units.fs,
        pfactor=2e6 * units.GPa * (units.fs ** 2),  # Barostat parameter in GPa
        num_chains=5,
        Q=150.0,
        adjoint=True,  
        topology_update_freq=1,
        mask=None
    ).to(system.device)
    if integrator.adjoint:
        integrator.adjoint_debug = ADJOINT_DEBUG
        integrator.npt_augmented_rhs = build_npt_augmented_rhs(
            integrator,
            debug=ADJOINT_DEBUG
        )

    

# define simulator with
sim = Simulations(system, integrator, method=solver_method)
# observe value
bins, g_obs, obs = get_observer(system, T, nbins)

if g_obs.device.type == 'cpu':
    g_obs = g_obs.to(device)
if not isinstance(bins, torch.Tensor):
    bins = torch.tensor(bins).to(device)
elif bins.device.type == 'cpu':
    bins = bins.to(device)

optimizer = torch.optim.Adam(list(net.parameters()), lr=params['lr'])
last_loss = float('inf')
loss_log = []
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                              'min',
                                              min_lr=1e-7,
                                              factor = 0.7, patience=15,
                                              threshold=1e-4)

for i in range(0, epochs):
    loss_js = torch.Tensor([0.0]).to(device)
    loss_mse = torch.Tensor([0.0]).to(device)
    if solver_method == "NH_verlet":
        v_t, q_t, pv_t = sim.simulate(steps=opt_freq, frequency=int(opt_freq))
    else:
        trajectories = sim.simulate(steps=opt_freq, frequency=int(opt_freq))
        if len(trajectories) == 6:
            v_t, q_t, pv_t, eta_t, zeta_t, h_t = trajectories
        else:
            v_t, q_t, pv_t = trajectories[:3]
            eta_t, zeta_t, h_t = None, None, None
    if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
        print("Have Nan value")
        exit(1)
    g, bins = obs(q_t[100::20])

    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
    if not isinstance(bins, torch.Tensor):
        bins = torch.tensor(bins)
    if g.device.type == 'cpu':
        g = g.to(device)
    if bins.device.type == 'cpu':
        bins = bins.to(device)
    
    loss_js += JS_rdf(g_obs, g)
    # loss_mse += assignments['mse_weight'] * (g - g_target_list[j]).pow(2).mean()
    rrange = torch.linspace(bins[0], bins[-1], g.shape[0]).to(device)
    rho = system.get_number_of_atoms() / system.get_volume()
    rho = torch.tensor(rho).to(device)
    loss_mse += compute_D(g - g_obs, rho, rrange)

    if i % 1 == 0:
        plot_rdfs(bins, g_obs, g, "{}_{}".format(T, i), model_path, pname=i, loss=loss_mse)

    loss = loss_mse
    optimizer.zero_grad()
    loss.backward()
    print("epoch {} | loss: {:.5f}".format(i, loss.item()))
    optimizer.step()
    scheduler.step(loss)
    loss_log.append(loss_js.item())

    if loss.item() < last_loss:
        print(f"Saving model at epoch {i} with loss {loss.item():.5f}")
        torch.save(net.state_dict(), f"{model_path}/net_best.pth")
        torch.save(prior.state_dict(), f"{model_path}/prior_best.pth")
        system.write(f"{model_path}/structure_epoch_{i}.cif")
        last_loss = loss.item()

plt.plot(loss_log)
plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
plt.close()