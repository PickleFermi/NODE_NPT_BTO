"""Summary
"""
import torch
import torchmd
from torchmd.util import GaussianSmearing
import numpy as np
from torchmd.system import check_system
from torchmd.topology import generate_nbr_list, get_offsets, generate_angle_list

def generate_vol_bins(start, end, bins, dim):
    
    # compute volume differential 
    if dim == 3:
        Vbins = 4 * np.pi /3*(bins[1:]**3 - bins[:-1]**3)
        V = (4/3)* np.pi * (end) ** 3
    elif dim == 2:
        Vbins = np.pi * (bins[1:]**2 - bins[:-1]**2)
        V = np.pi * (end) ** 2
        
    return V, torch.Tensor(Vbins), bins


class Observable(torch.nn.Module):
    def __init__(self, system):
        super(Observable, self).__init__()
        check_system(system)
        self.device = system.device
        self.volume = system.get_volume()
        self.cell = torch.Tensor( system.get_cell()).diag().to(self.device)
        self.natoms = system.get_number_of_atoms()


def differentiable_histogram(distances, bin_edges):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Compute bin width
    bin_width = bin_edges[1] - bin_edges[0]
    diff = distances[:, None] - bin_centers[None, :]
    sigma = bin_width / np.sqrt(2 * np.pi) # np.sqrt(2 * np.pi)
    K = torch.exp(-0.5 * (diff / sigma) ** 2)
    counts = K.sum(dim=0)
    return counts


class pdf(Observable):
    def __init__(self, system, nbins, bins, r_range, index_tuple=None, width=None):
        super(pdf, self).__init__(system)
        PI = np.pi

        self.numbers = torch.tensor(system.numbers).to(self.device)
        # Default coefficients for neutron PDF weights.
        # Keep legacy Pb/Zr entries and add Ba counterparts for BaTiO3 systems.
        self.pair_coeffi = {
            (82, 82): 125.44,  # Pb-Pb
            (82, 22): 98.56,   # Pb-Ti
            (82, 8): 107.52,   # Pb-O
            (82, 40): 98.56,   # Pb-Zr
            (56, 56): 125.44,  # Ba-Ba
            (56, 22): 98.56,   # Ba-Ti
            (56, 8): 107.52,   # Ba-O
            (56, 40): 98.56,   # Ba-Zr
            (22, 22): 19.36,   # Ti-Ti
            (22, 8): 42.24,    # Ti-O
            (22, 40): 19.36,   # Ti-Zr
            (40, 40): 19.36,   # Zr-Zr
            (40, 8): 42.24,    # Zr-O
            (40, 22): 19.36,   # Zr-Ti
            (8, 8): 23.04,     # O-O
        }
        unique_numbers = sorted(torch.unique(self.numbers).tolist())
        # Build pair list from actual species in this system so all neighbor
        # pairs are covered (and assertion in forward remains valid).
        self.atom_pairs = []
        for i, z1 in enumerate(unique_numbers):
            for z2 in unique_numbers[i:]:
                self.atom_pairs.append((int(z1), int(z2)))
        start = r_range[0]
        end = r_range[1]
        self.device = system.device
        V, vol_bins, bins = generate_vol_bins(start, end, bins, dim=system.dim)

        self.V = V
        self.vol_bins = vol_bins.to(self.device)
        self.bins = torch.tensor(bins, device=self.device, dtype=torch.float32)

        self.nbins = nbins
        self.cutoff_boundary = end + 5e-1
        self.index_tuple = index_tuple

    def forward(self, xyz):
        if xyz.dim() == 3:
            xyz_l = [i for i in xyz]
        else:
            xyz_l = [xyz]

        pdf_l = []
        for xyz in xyz_l:
            nbr_list, pair_dis, _ = generate_nbr_list(xyz,
                                                      self.cutoff_boundary,
                                                      self.cell,
                                                      index_tuple=self.index_tuple,
                                                      get_dis=True)
            # nbr_list = nbr_list[:, -2:] # 若有多个样本
            number_list = self.numbers[nbr_list]
            pdf = 0
            pair_count = 0
            for pair in self.atom_pairs:
                pair_tensor = torch.tensor(pair).to(self.device)
                pair_tensor_T = torch.tensor([pair[1],pair[0]]).to(self.device)
                mask1 = torch.all(number_list == pair_tensor, dim=-1)
                mask2 = torch.all(number_list == pair_tensor_T, dim=-1)
                mask = mask1 + mask2
                pair_count += len(torch.where(mask==True)[0])

                selected_pairs = pair_dis[mask].reshape(-1)
                count = differentiable_histogram(selected_pairs, self.bins)
                norm = count.sum()  # normalization factor for histogram
                count = count / norm  # normalize
                rdf = count / (self.vol_bins / self.V)
                coeff = self.pair_coeffi.get(pair, self.pair_coeffi.get((pair[1], pair[0]), 1.0))
                pdf += coeff * (rdf - 1)
            assert pair_count == len(number_list)
            pdf_l.append(pdf)
        pdf_l = torch.stack(pdf_l, dim=0)
        pdf_average = torch.mean(pdf_l, dim=0) # ensemble average
        pdf_average = pdf_average / 416.16
        return pdf_average, self.bins

class Angles(Observable):
    def __init__(self, system, nbins, angle_range, cutoff=3.0,index_tuple=None, width=None):
        super(Angles, self).__init__(system)
        PI = np.pi
        start = angle_range[0]
        end = angle_range[1]
        self.device = system.device
        self.bins = torch.linspace(start, end, nbins + 1).to(self.device)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        ).to(self.device)
        self.width = (self.smear.width[0]).item()
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        
    def forward(self, xyz):
        
        xyz = xyz.reshape(-1, self.natoms, 3)

        nbr_list, _ = generate_nbr_list(xyz, self.cutoff,
                                           self.cell, 
                                           index_tuple=self.index_tuple, 
                                           get_dis=False)
        nbr_list = nbr_list.to("cpu")
        
        angle_list = generate_angle_list(nbr_list).to(self.device)
        cos_angles = compute_angle(xyz, angle_list, self.cell, N=self.natoms)

        return cos_angles

class angle_distribution(Observable):
    def __init__(self, system, nbins, angle_range, cutoff=3.0,index_tuple=None, width=None):
        super(angle_distribution, self).__init__(system)
        PI = np.pi
        start = angle_range[0]
        end = angle_range[1]
        self.device = system.device
        self.bins = torch.linspace(start, end, nbins + 1).to(self.device)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        ).to(self.device)
        self.width = (self.smear.width[0]).item()
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        
    def forward(self, xyz):
        
        xyz = xyz.reshape(-1, self.natoms, 3)

        nbr_list, _ = generate_nbr_list(xyz, self.cutoff,
                                           self.cell, 
                                           index_tuple=self.index_tuple, 
                                           get_dis=False)
        nbr_list = nbr_list.to("cpu")
        
        angle_list = generate_angle_list(nbr_list).to(self.device)
        cos_angles = compute_angle(xyz, angle_list, self.cell, N=self.natoms)
        
        angles = cos_angles.acos()

        count = self.smear(angles.reshape(-1).squeeze()[..., None]).sum(0) 

        norm = count.sum()   # normalization factor for histogram 
        count = count / (norm)  # normalize 
        
        return self.bins, count, angles

class vacf(Observable):
    def __init__(self, system, t_range):
        super(vacf, self).__init__(system)
        self.t_window = [i for i in range(1, t_range, 1)]

    def forward(self, vel):
        vacf = [(vel * vel).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in self.t_window]

        return torch.stack(vacf).reshape(-1)


def compute_angle(xyz, angle_list, cell, N):
    
    device = xyz.device
    xyz = xyz.reshape(-1, N, 3)
    bond_vec1 = xyz[angle_list[:,0], angle_list[:,1]] - xyz[angle_list[:,0], angle_list[:, 2]]
    bond_vec2 = xyz[angle_list[:,0], angle_list[:,3]] - xyz[angle_list[:,0], angle_list[:, 2]]
    bond_vec1 = bond_vec1 + get_offsets(bond_vec1, cell, device) * cell
    bond_vec2 = bond_vec2 + get_offsets(bond_vec2, cell, device) * cell  
    
    angle_dot = (bond_vec1 * bond_vec2).sum(-1)
    norm = ( bond_vec1.pow(2).sum(-1) * bond_vec2.pow(2).sum(-1) ).sqrt()
    cos = angle_dot / norm
    
    return cos

def compute_dihe(xyz, dihes): 
    assert len(xyz.shape) == 3
    n_frames = xyz.shape[0]
    N = xyz.shape[1]
    xyz = xyz[:, :, None, :]
    D = xyz.expand(n_frames, N,N,3)-xyz.expand(n_frames, N,N,3).transpose(1,2)
    vec1 = D[:, dihes[:,1], dihes[:,0]]
    vec2 = D[:, dihes[:,1], dihes[:,2]]
    vec3 = D[:, dihes[:,2], dihes[:,1]]
    vec4 = D[:, dihes[:,2], dihes[:,3]]
    cross1 = torch.cross(vec1, vec2)
    cross2 = torch.cross(vec3, vec4)

    norm = (cross1.pow(2).sum(-1)*cross2.pow(2).sum(-1)).sqrt()
    cos_phi = 1.0*((cross1*cross2).sum(-1)/norm)
    
    return cos_phi 


# New Observable function packed with data and plotting 

# class Observable(torch.snn.Module):
#     def __init__(self, system, tag, data, loss_func):
#         super(Observable, self).__init__()
#         #check_system(system)
#         self.device = system.device
#         self.volume = system.get_volume()
#         self.cell = torch.Tensor( system.get_cell()).diag().to(self.device)
#         self.natoms = system.get_number_of_atoms()

#         self.tag = tag 
#         self.data = data.to(system.device) # n-dim array
#         self.loss_func = loss_func
#         self.log = []
    
#     def get_loss(self, x):
#         return self.loss_func(self(x), self.data) 
    
#     def update_log(self, results):
#         self.log.append(results.detach().cpu().numpy())
        
#     def save_log(self, path):
#         np.savetxt(path, np.array(self.log))
    
#     def forward(self):
#         pass 
        

# class rdf(Observable):
#     def __init__(self, 
#                  system, 
#                  nbins, 
#                  r_range, 
#                  data,
#                  tag,
#                  loss_func,
#                  index_tuple=None, 
#                  width=None):
        
#         super(rdf, self).__init__(system, tag, data, loss_func)
#         PI = np.pi

#         start = r_range[0]
#         end = r_range[1]
#         self.device = system.device

#         V, vol_bins, bins = generate_vol_bins(start, end, nbins, dim=system.dim)

#         self.V = V
#         self.vol_bins = vol_bins.to(self.device)
#         self.r_axis = np.linspace(start, end, nbins)
#         self.device = system.device
#         self.bins = bins

#         self.smear = GaussianSmearing(
#             start=start,
#             stop=bins[-1],
#             n_gaussians=nbins,
#             width=width,
#             trainable=False
#         ).to(self.device)

#         self.nbins = nbins
#         self.cutoff_boundary = end + 5e-1
#         self.index_tuple = index_tuple
        
#     def forward(self, xyz):

#         nbr_list, pair_dis, _ = generate_nbr_list(xyz, 
#                                                self.cutoff_boundary, 
#                                                self.cell, 
#                                                index_tuple=self.index_tuple, 
#                                                get_dis=True)

#         count = self.smear(pair_dis.reshape(-1).squeeze()[..., None]).sum(0) 
#         norm = count.sum()   # normalization factor for histogram 
#         count = count / norm   # normalize 
#         count = count
#         rdf =  count / (self.vol_bins / self.V ) 
        
#         self.update_log(rdf)
#         return rdf
    
#     def plot(self, fname, save=False):
        
#         plt.plot(self.r_axis, 
#                  self.log[-1], 
#                  label='simulation', 
#                  linewidth=4, alpha=0.6)
        
#         plt.plot(self.r_axis,
#                  self.data.detach().cpu().numpy(), 
#                  label='target', 
#                  linewidth=2, linestyle='--', c='black')

#         plt.xlabel("$\AA$")
#         plt.ylabel("g(r)")
        
#         plt.show()
        
#         if save:
#             plt.savefig(fname, bbox_inches='tight')
            
#         plt.close()


# class vacf(Observable):
#     def __init__(self, system, t_range, tag, data, loss_func):
#         super(vacf, self).__init__(system, tag, data, loss_func)
#         self.t_window = [i for i in range(1, t_range, 1)]
#         self.t_lim = [i for i in range(0, t_range, 1)]

#     def forward(self, vel):
#         vacf = [(vel * vel).mean()[None]]
#         # can be implemented in parrallel
#         vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in self.t_window]
#         vacf = torch.stack(vacf).reshape(-1)
#         self.update_log(vacf)

#         return vacf
    
#     def plot(self, fname, save=False):
        
#         plt.plot(self.t_lim, self.log[-1], label='simulation', linewidth=4, alpha=0.6, )

#         if self.data is not None:
#             plt.plot(self.t_lim, self.data.detach().cpu().numpy(), label='target', linewidth=2,linestyle='--', c='black' )

#         plt.legend()

#         if save:
#             plt.savefig(fname, bbox_inches='tight')
#         plt.show()
#         plt.close()

# from scripts.data import exp_rdf_data_dict, pair_data_dict, get_exp_rdf
# from torchmd.observable import generate_vol_bins

# rdfparams = {"system": system,
#  "nbins": 100,
#  "r_range": (0.25, 7.5)}

# tag ='H20_298K_redd'

# def L2(x1, x2):
#     return (x1 - x2).pow(2).mean()

# # example rdf data 
# rdfdata = np.loadtxt(exp_rdf_data_dict[tag]['fn'], delimiter=',')
# rdfdata = get_exp_rdf(rdfdata, 100, (0.25, 7.5), device=system.device)

# # example vacf data

# vacfparams = {'system': system, 
#               't_range': 60}

# vacfdata = np.loadtxt(pair_data_dict['lj_0.9_1.2']['vacf_fn'])
# t_axis =  np.linspace(0.0,  vacfdata.shape[0], vacfdata.shape[0] ) * 0.01
# vacfdata = np.transpose( np.concatenate((t_axis, vacfdata)).reshape(2, -1) )

# obs = rdf(system, nbins=100, r_range=(0.25, 8), 
#           data=rdfdata[1], loss_func=L2, tag=tag) 


# vacf_obs = vacf(system, t_range=60, 
#                 data=torch.Tensor(vacfdata[:, 1]), 
#                 tag='lj_0.9_1.2', 
#                 loss_func=L2)

# obs.get_loss(q_t[::10])
