import torch
import numpy as np
import itertools

def compute_dis(xyz, nbr_list, offsets, cell): 
    '''
        compute distance give nbr_list and periodic boundary condition
    '''
    e = (xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]] -
             offsets.matmul(cell)).pow(2).sum(1).sqrt()[:, None]

    return e


def generate_pair_index(N, index_tuple, device=None):

    if device is None:
        mask_sel = torch.zeros(N, N)
    else:
        mask_sel = torch.zeros(N, N, device=device)

    if index_tuple is not None:
        if device is None:
            pair_mask = torch.LongTensor( [list(items) for items in itertools.product(index_tuple[0], 
                                                                                      index_tuple[1])]) 
        else:
            pair_mask = torch.LongTensor( [list(items) for items in itertools.product(index_tuple[0], 
                                                                                      index_tuple[1])]).to(device) 

        #todo: imporse index convention
        mask_sel[pair_mask[:, 0], pair_mask[:, 1]] = 1
        mask_sel[pair_mask[:, 1], pair_mask[:, 0]] = 1
    
    return mask_sel


def generate_nbr_list(xyz, cutoff, cell, index_tuple=None, ex_pairs=None, get_dis=False):
    
    # todo: topology should be a class to handle some initialization 
    device = xyz.device

    dis_mat = (xyz[..., None, :, :] - xyz[..., :, None, :])

    if index_tuple is not None:
        N = xyz.shape[-2] # the 2nd dim is the atoms dim

        mask_sel = generate_pair_index(N, index_tuple, device if device != 'cpu' else None)
        if device != 'cpu' and mask_sel.device.type == 'cpu':
            mask_sel = mask_sel.to(device)
        # todo handle this calculation like a sparse tensor 
        dis_mat =  dis_mat * mask_sel[..., None]

    if ex_pairs is not None:

        N = xyz.shape[-2] # the 2nd dim is the atoms dim

        mask = torch.ones(N, N)
        mask[ex_pairs[:,0], ex_pairs[:, 1]] = 0
        mask[ex_pairs[:,1], ex_pairs[:, 0]] = 0

        # todo handle this calculation like a sparse tensor 
        if device != 'cpu':
            mask = mask.to(device)
        dis_mat = dis_mat * mask[..., None]
        
    if len(cell.shape) == 1: # I should probably completely get rid of cubic cell convention, will do it some other day 
        cell = torch.diag(cell)
    
    # project the position vector onto the cell basis (does not need to be orthonormal )
    reduced_dis = dis_mat.matmul( cell.inverse())
    threshold = torch.tensor([0.5, 0.5, 0.5], device=device)
    offsets_add = -(reduced_dis > threshold).to(torch.float)
    offsets_sub = (reduced_dis < -threshold).to(torch.float)
    offsets = offsets_add + offsets_sub
    
    dis_mat = dis_mat + offsets.matmul(cell)
    
    dis_sq = torch.triu( dis_mat.pow(2).sum(-1) )
    mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
    nbr_list = torch.nonzero( torch.triu(mask.to(torch.long)), as_tuple=False)

    if get_dis:
        return nbr_list, dis_sq[mask].sqrt(), offsets[nbr_list[:, 0], nbr_list[:, 1], :]
    else:
        return nbr_list, offsets[nbr_list[:, 0], nbr_list[:, 1], :]

def get_offsets(vecs, cell, device):
    
    offsets = -vecs.ge(0.5 *  cell).to(torch.float) + \
                vecs.lt(-0.5 *  cell).to(torch.float)
    
    return offsets


def generate_angle_list(nbr_list):
    '''
        Contributed by saxelrod
    '''

    assert nbr_list.shape[1] == 3 

    nbr_list = make_directed(nbr_list)

    mask = (nbr_list[:, 2, None] == nbr_list[:, 1]) * (
            nbr_list[:, 1, None] != nbr_list[:, 2]) * ( 
            nbr_list[:, None, 0] == nbr_list[:, 0]) # select the same frame 

    third_atoms = nbr_list[:, 2].repeat(nbr_list.shape[0], 1)[mask]
    num_angles = mask.sum(-1)

    if nbr_list.device.type != 'cpu':
        nbr_repeats = torch.LongTensor(
                    np.repeat(nbr_list.cpu().numpy(), num_angles.cpu().tolist(), axis=0)).to(nbr_list.device)
    else:
        nbr_repeats = torch.LongTensor(
                    np.repeat(nbr_list.numpy(), num_angles.tolist(), axis=0))

    angle_list = torch.cat(
        (nbr_repeats, third_atoms.reshape(-1, 1)), dim=1)
    
    return angle_list 

def make_directed(nbr_list):
    """
    Check if a neighbor list is directed, and make it
    directed if it isn't. Contributed by saxelrod
    Args:
        nbr_list (torch.LongTensor): neighbor list
    Returns:
        new_nbrs (torch.LongTensor): directed neighbor
            list
        directed (bool): whether the old one was directed
            or not  
    """
    nbr_list_reverse = torch.stack([nbr_list[:,0], nbr_list[:, 2], nbr_list[:,1]], dim=-1)
    new_nbrs = torch.cat([nbr_list, nbr_list_reverse], dim=0)
    
    return new_nbrs



if __name__ == "__main__": 

    from ase.lattice.cubic import FaceCenteredCubic

    atoms = FaceCenteredCubic(symbol='H',
                          size=(3, 3, 3),
                          latticeconstant=1.679,
                          pbc=True)

    from ase.neighborlist import neighbor_list


    i, j, d= neighbor_list("ijD", atoms, cutoff=2.5, self_interaction=False)

    print("ASE calculated {} pairs".format(i.shape[0]))
    xyz = torch.Tensor( atoms.get_positions() )
    cell = torch.Tensor(atoms.get_cell() )
    cutoff = 2.5

    nbr_list, offsets = generate_nbr_list(xyz, cutoff, cell)

    print("torchmd calculated {} pairs".format(nbr_list.shape[0] * 2 ))