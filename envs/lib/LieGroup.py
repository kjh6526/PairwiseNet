import torch
import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2
angle_threshold = 1e-6

def tn_converter(x):
    if type(x) == type(np.zeros(1)):
        x = torch.tensor(x).unsqueeze(0)
    elif type(x) == type(torch.tensor(1)):
        x = x.squeeze(0).detach().cpu().numpy()
    return x

def getNullspace(tensor2dim):
    if tensor2dim.is_complex():
        print('ERROR : getNullspace() fed by a complex number')
        exit(1)
    U, S, V = torch.Tensor.svd(tensor2dim, some=False, compute_uv=True)
    # threshold = torch.max(S) * torch.finfo(S.dtype).eps * max(U.shape[0], V.shape[1])
    # rank = torch.sum(S > threshold, dtype=int)
    rank = len(S)
    # return V[rank:, :].T.cpu().conj()
    return V[:, rank:]


def revoluteTwist(twist):
    nJoint, mustbe6 = twist.shape
    if mustbe6 != 6:
        print(f'[ERROR] revoluteTwist: twist.shape = {twist.shape}')
        exit(1)
    w = twist[:, :3]
    v = twist[:, 3:]
    w_normalized = w / w.norm(dim=1).view(nJoint, 1)
    proejctedTwists = torch.empty_like(twist)
    proejctedTwists[:, :3] = w_normalized
    wdotv = torch.sum(v * w_normalized, dim=1).view(nJoint, 1)
    proejctedTwists[:, 3:] = v - wdotv * w_normalized
    return proejctedTwists


def skew3dim(vec3dim):
    return skew_so3(vec3dim)


def skew_so3(so3):
    nBatch = len(so3)
    if so3.shape == (nBatch, 3, 3):
        return torch.cat([-so3[:, 1, 2].unsqueeze(-1),
                          so3[:, 0, 2].unsqueeze(-1),
                          -so3[:, 0, 1].unsqueeze(-1)], dim=1)
    elif so3.numel() == nBatch * 3:
        w = so3.reshape(nBatch, 3, 1, 1)
        zeroBatch = so3.new_zeros(nBatch, 1, 1)
        output = torch.cat([torch.cat([zeroBatch, -w[:, 2], w[:, 1]], dim=2),
                            torch.cat([w[:, 2], zeroBatch, -w[:, 0]], dim=2),
                            torch.cat([-w[:, 1], w[:, 0], zeroBatch], dim=2)], dim=1)
        return output
    else:
        print(f'ERROR : skew_so3, so3.shape = {so3.shape}')
        exit(1)


def skew6dim(vec6dim):
    return skew_se3(vec6dim)


def skew_se3(se3):
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        output = se3.new_zeros(nBatch, 6)
        output[:, :3] = skew_so3(se3[:, :3, :3])
        output[:, 3:] = se3[:, :3, 3]
        return output
    elif se3.numel() == nBatch * 6:
        se3_ = se3.reshape(nBatch, 6)
        output = se3_.new_zeros(nBatch, 4, 4)
        output[:, :3, :3] = skew_so3(se3_[:, :3])
        output[:, :3, 3] = se3_[:, 3:]
        return output
    else:
        print(f'ERROR : skew_se3, se3.shape = {se3.shape}')
        exit(1)


def expSO3(so3):
    nBatch = len(so3)
    if so3.shape == (nBatch, 3, 3):
        so3mat = so3
        so3vec = skew_so3(so3)
    elif so3.numel() == nBatch * 3:
        so3mat = skew_so3(so3)
        so3vec = so3.reshape(nBatch, 3)
    else:
        print(f'ERROR : expSO3, so3.shape = {so3.shape}')
        exit(1)
    # Rodrigues' rotation formula
    theta = so3vec.norm(dim=1).view(nBatch, 1)
    wmat = skew_so3(so3vec / theta)
    expso3 = so3.new_zeros(nBatch, 3, 3)
    expso3[:, 0, 0] = expso3[:, 1, 1] = expso3[:, 2, 2] = 1
    expso3 += theta.sin().view(nBatch, 1, 1) * wmat
    expso3 += (1 - theta.cos()).view(nBatch, 1, 1) * wmat @ wmat
    return expso3


def expSE3(se3):
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        se3mat = se3
        se3vec = skew_se3(se3)
    elif se3.numel() == nBatch * 6:
        se3mat = skew_se3(se3)
        se3vec = se3.reshape(nBatch, 6)
    else:
        print(f'ERROR : expSE3, se3.shape = {se3.shape}')
        exit(1)
    # normalize
    w = se3vec[:, :3]
    v = se3vec[:, 3:]
    theta = w.norm(dim=1)
    zeroID = theta == 0
    expse3 = se3.new_zeros(nBatch, 4, 4)
    # wmat = skew_so3(w / theta.view(nBatch,1))
    # # G = eye * theta + (1-cos(theta)) [w] + (theta - sin(theta)) * [w]^2
    # G = se3.new_zeros(nBatch, 3, 3)
    # G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = theta.view(nBatch)
    # G += (1 - theta.cos()).view(nBatch, 1, 1) * wmat
    # G += (theta - theta.sin()).view(nBatch, 1, 1) * wmat @ wmat
    # # output
    # expse3[:, :3, :3] = expSO3(w)
    # expse3[:, :3, 3] = (G @ (v / theta.view(nBatch,1) ).view(nBatch,3,1)).view(nBatch, 3)
    # expse3[:, 3, 3] = 1
    if zeroID.any():
        expse3[zeroID, 0, 0] = expse3[zeroID, 1, 1] = expse3[zeroID, 2, 2] = expse3[zeroID, 3, 3] = 1
        expse3[zeroID, :3, 3] = v[zeroID]
    if (~zeroID).any():
        nNonZero = (~zeroID).sum()
        _theta = theta[~zeroID].reshape(nNonZero, 1)
        wmat = skew_so3(w[~zeroID] / _theta)
        # G = eye * theta + (1-cos(theta)) [w] + (theta - sin(theta)) * [w]^2
        G = se3.new_zeros(nNonZero, 3, 3)
        G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = _theta.view(nNonZero)
        G += (1 - _theta.cos()).view(nNonZero, 1, 1) * wmat
        G += (_theta - _theta.sin()).view(nNonZero, 1, 1) * wmat @ wmat
        # output
        expse3[~zeroID, :3, :3] = expSO3(w[~zeroID])
        expse3[~zeroID, :3, 3] = (G @ (v[~zeroID] / _theta).view(nNonZero, 3, 1)).view(nNonZero, 3)
        expse3[~zeroID, 3, 3] = 1
    return expse3
    # return torch.matrix_exp(se3mat)

def clipping(x, low=-1.0, high=1.0):
    eps = 1e-6
    x[x<=low] = low + eps
    x[x>=high] = high - eps
    return x

def logSO3(SO3):
    nBatch = len(SO3)
    trace = torch.einsum('xii->x', SO3)
    regularID = (trace + 1).abs() >= angle_threshold
    singularID = (trace + 1).abs() < angle_threshold
    theta = torch.acos(clipping((trace - 1) / 2)).view(nBatch, 1, 1)
    so3mat = SO3.new_zeros(nBatch, 3, 3)
    # regular
    if any(regularID):
        so3mat[regularID, :, :] = (SO3[regularID] - SO3[regularID].transpose(1, 2)) / (2 * theta[regularID].sin()) * theta[regularID]
    # singular
    if any(singularID):
        if all(SO3[singularID, 2, 2] != -1):
            r = SO3[singularID, 2, 2]
            w = SO3[singularID, :, 2]
            w[:, 2] += 1
        elif all(SO3[singularID, 1, 1] != -1):
            r = SO3[singularID, 1, 1]
            w = SO3[singularID, :, 1]
            w[:, 1] += 1
        elif all(SO3[singularID, 0, 0] != -1):
            r = SO3[singularID, 0, 0]
            w = SO3[singularID, :, 0]
            w[:, 0] += 1
        else:
            print(f'ERROR: all() is somewhat ad-hoc. should be fixed.')
            exit(1)
        so3mat[singularID, :, :] = skew_so3(torch.pi / (2 * (1 + r)).sqrt().view(-1, 1) * w)
    # trace == 3 (zero rotation)
    if any((trace - 3).abs() < 1e-10):
        so3mat[(trace - 3).abs() < 1e-10] = 0
    return so3mat

def logSO3_v2(R):
    batch_size = R.shape[0]
    eps = 1e-4
    trace = torch.sum(R[:, range(3), range(3)], dim=1)

    omega = R * torch.zeros(R.shape).to(R)

    theta = torch.acos(torch.clip((trace - 1) / 2, min=-1+1e-6, max=1-1e-6))

    temp = theta.unsqueeze(-1).unsqueeze(-1)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew_so3(omega_vector) * torch.pi

    return omega

def logSE3(SE3):
    nBatch = len(SE3)
    trace = torch.einsum('xii->x', SE3[:, :3, :3])
    regularID = (trace - 3).abs() >= angle_threshold
    zeroID = (trace - 3).abs() < angle_threshold
    se3mat = SE3.new_zeros(nBatch, 4, 4)
    if any(zeroID):
        se3mat[zeroID, :3, 3] = SE3[zeroID, :3, 3]
    if any(regularID):
        nRegular = sum(regularID)
        so3 = logSO3(SE3[regularID, :3, :3])
        # theta = skew_so3(so3).norm(dim=1).reshape(nRegular,1,1)
        theta = (torch.acos(clipping(0.5*(trace[regularID]-1)))).reshape(nRegular, 1, 1)
        wmat = so3 / theta
        identity33 = torch.zeros_like(so3)
        identity33[:, 0, 0] = identity33[:, 1, 1] = identity33[:, 2, 2] = 1
        invG = (1 / theta) * identity33 - 0.5 * wmat + (1 / theta - 0.5 / (0.5 * theta).tan()) * wmat @ wmat
        se3mat[regularID, :3, :3] = so3
        se3mat[regularID, :3, 3] = theta.view(nRegular, 1) * (invG @ SE3[regularID, :3, 3].view(nRegular, 3, 1)).reshape(nRegular, 3)
    return se3mat


def largeAdjoint(SE3):
    # R     0
    # [p]R  R
    nBatch = len(SE3)
    if SE3.shape != (nBatch, 4, 4):
        print(f'ERROR : SE3.shape = {SE3.shape}')
        exit(1)
    R, p = SE3[:, :3, :3], SE3[:, :3, 3].unsqueeze(-1)
    Adj = SE3.new_zeros(nBatch, 6, 6)
    Adj[:, :3, :3] = Adj[:, 3:, 3:] = R
    Adj[:, 3:, :3] = skew_so3(p) @ R
    return Adj


def smallAdjoint(se3):
    # [w] 0
    # [v] [w]
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        se3vec = skew_se3(se3)
    elif se3.numel() == nBatch * 6:
        se3vec = se3.reshape(nBatch, 6)
    else:
        print(f'ERROR : smallAdjoint, se3.shape = {se3.shape}')
        exit(1)
    w = se3vec[:, :3]
    v = se3vec[:, 3:]
    smallad = se3.new_zeros(nBatch, 6, 6)
    smallad[:, :3, :3] = smallad[:, 3:, 3:] = skew_so3(w)
    smallad[:, 3:, :3] = skew_so3(v)
    return smallad


def invSE3(SE3):
    nBatch = len(SE3)
    R, p = SE3[:, :3, :3], SE3[:, :3, 3].unsqueeze(-1)
    invSE3_ = SE3.new_zeros(nBatch, 4, 4)
    invSE3_[:, :3, :3] = R.transpose(1, 2)
    invSE3_[:, :3, 3] = - (R.transpose(1, 2) @ p).view(nBatch, 3)
    invSE3_[:, 3, 3] = 1
    return invSE3_


def invSO3(SO3):
    nBatch = len(SO3)
    if SO3.shape != (nBatch, 3, 3):
        print(f'[ERROR] invSO3 : SO3.shape = {SO3.shape}')
        exit(1)
    return SO3.transpose(1, 2)

def SO3_to_quatonian(R, ordering='wxyz'):
    qw = 0.5 * torch.sqrt(1. + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    qx = qw.new_empty(qw.shape)
    qy = qw.new_empty(qw.shape)
    qz = qw.new_empty(qw.shape)

    near_zero_mask = qw.abs_().lt(1e-6)

    if sum(near_zero_mask) > 0:
        cond1_mask = near_zero_mask & \
            (R[:, 0, 0] > R[:, 1, 1]).squeeze_() & \
            (R[:, 0, 0] > R[:, 2, 2]).squeeze_()
        cond1_inds = cond1_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(cond1_inds) > 0:
            R_cond1 = R[cond1_inds]
            d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] -
                                R_cond1[:, 1, 1] - R_cond1[:, 2, 2])
            qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
            qx[cond1_inds] = 0.25 * d
            qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
            qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

        cond2_mask = near_zero_mask & (R[:, 1, 1] > R[:, 2, 2]).squeeze_()
        cond2_inds = cond2_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(cond2_inds) > 0:
            R_cond2 = R[cond2_inds]
            d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] -
                                R_cond2[:, 0, 0] - R_cond2[:, 2, 2])
            qw[cond2_inds] = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
            qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
            qy[cond2_inds] = 0.25 * d
            qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

        cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
        cond3_inds = cond3_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(cond3_inds) > 0:
            R_cond3 = R[cond3_inds]
            d = 2. * \
                torch.sqrt(1. + R_cond3[:, 2, 2] -
                        R_cond3[:, 0, 0] - R_cond3[:, 1, 1])
            qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
            qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
            qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
            qz[cond3_inds] = 0.25 * d

    far_zero_mask = near_zero_mask.logical_not()
    far_zero_inds = far_zero_mask.nonzero(as_tuple=False).squeeze_(dim=1)
    if len(far_zero_inds) > 0:
        R_fz = R[far_zero_inds]
        d = 4. * qw[far_zero_inds]
        qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
        qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
        qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

    # Check ordering last
    if ordering is 'xyzw':
        quat = torch.cat([qx.unsqueeze_(dim=1),
                            qy.unsqueeze_(dim=1),
                            qz.unsqueeze_(dim=1),
                            qw.unsqueeze_(dim=1)], dim=1).squeeze_()
    elif ordering is 'wxyz':
        quat = torch.cat([qw.unsqueeze_(dim=1),
                            qx.unsqueeze_(dim=1),
                            qy.unsqueeze_(dim=1),
                            qz.unsqueeze_(dim=1)], dim=1).squeeze_()
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    return quat