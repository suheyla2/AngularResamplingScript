# This code only works for single bvalue shells. If you have multiple bvalue shells, you need to extract the single bval and run this code on each bvalue shell separately.
# Example usage:
# imgPath is the DWI path of the higher angular resolution data.
# maskPath is the mask of the DWI.
# outPrefix is the prefix of the output files (we save the max bval of the input DWI as the bval of the output DWI, and we use the bvecs of the lower angular resolution DWI to get the bvecs of the output DWI, and DWI data is the resampled DWI data of the higher angular resolution).
# imgLowerAngular is the DWI with lower angular resolution (we use this to get the bvals and bvecs of the lower angular resolution DWI).

# resampleDWIgradients(
#    imgPath="./testdata/HighAngularResData.nii.gz",
#    maskPath="./testdata/mask.nii.gz",
#    outPrefix="./testdata/ResampledData",
#    imgLowerAngular="./testdata/LowAngularResData.nii.gz"
# )
import os
import numpy as np
from nibabel import load
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import QballModel
from dipy.segment.mask import applymask
from os.path import isfile, join as pjoin, split as psplit, abspath, basename, dirname, exists
from os import makedirs, remove, listdir, getenv, getpid, chdir, path

def find_b0(dwi, where_b0, mask=None):
    b0 = dwi[..., where_b0].mean(-1)
    np.nan_to_num(b0).clip(min=0., out=b0)

    if mask is not None:
        return b0 * mask
    else:
        return b0

def normalize_data(dwi, where_b0=None, mask=None, b0=None):
    dwi = dwi.astype('float32')

    if where_b0 is not None and b0 is None:
        b0 = find_b0(dwi, where_b0, mask)
        np.nan_to_num(b0).clip(min=1., out=b0)  # can be changed to 0. as well
        for i in where_b0:
            dwi[..., i] = b0
    else:
        np.nan_to_num(b0).clip(min=1., out=b0)  # can be changed to 0. as well

    dwiPrime = dwi / b0[..., None]
    np.nan_to_num(dwiPrime).clip(min=0., max=1., out=dwiPrime)

    if mask is not None:
        dwiPrime = applymask(dwiPrime, mask)

    return (dwiPrime, b0)

def resampleDWIgradients(imgPath, maskPath, outPrefix, imgLowerAngular, qb_model=None, B0_THRESH=50., N_shm=8):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Current Working Directory:", os.getcwd())

    img = load(imgPath)
    data = img.get_fdata()
    affine = img.affine
    hdr = img.header
    mask_data = load(maskPath).get_fdata()

    inPrefix = imgPath.split('.nii')[0]
    prefix = psplit(inPrefix)[-1]
    inPrefixLowerAngular = imgLowerAngular.split('.nii')[0]

    if not qb_model:
        print('Computing shm_coeff of', imgPath)
        bvals, bvecs = read_bvals_bvecs(inPrefix + '.bval', inPrefix + '.bvec')
        bvals_low, bvecs_low = read_bvals_bvecs(inPrefixLowerAngular + '.bval', inPrefixLowerAngular + '.bvec')
        
        bvals = np.append(bvals, bvals)
        bvecs = np.append(bvecs, -bvecs, axis=0)
        bvals_low = np.append(bvals_low, -bvals_low)
        bvecs_low = np.append(bvecs_low, -bvecs_low, axis=0)
        data = np.append(data, data, axis=3)

        gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESH)
        gtab_low = gradient_table(bvals_low, bvecs_low, b0_threshold=B0_THRESH)
    
        qb_model = QballModel(gtab, sh_order_max=N_shm)
        qb_model_low = QballModel(gtab_low, sh_order_max=N_shm)

        b0 = find_b0(data, where_b0=np.where(qb_model.gtab.b0s_mask)[0])
        if not isfile(inPrefix + '_bse.nii.gz'):
            save_nifti(inPrefix + '_bse.nii.gz', applymask(b0, mask_data), affine, hdr)
    else:
        b0 = None

    smooth = 0.00001
    data = applymask(data, mask_data)
    data_norm, _ = normalize_data(data, where_b0=np.where(qb_model.gtab.b0s_mask)[0])
    print('shape of data_norm:', data_norm.shape)

    dwi_indices = np.where(~qb_model.gtab.b0s_mask)[0]
    data_norm_dwi = data_norm[..., dwi_indices]
    print('shape of data_norm_dwi:', data_norm_dwi.shape)

    L = -qb_model.l_values * (qb_model.l_values + 1)
    L **= 2
    _fit_matrix = np.linalg.pinv(qb_model.B.T @ qb_model.B + np.diag(smooth * L)) @ qb_model.B.T
    shm_coeff = np.dot(data_norm[..., qb_model._where_dwi], _fit_matrix.T)
    shm_coeff = applymask(shm_coeff, mask_data)

    shm_coeff_reshaped = shm_coeff.reshape(-1, shm_coeff.shape[-1])
    estimatedSignal_reshaped = qb_model_low.B @ shm_coeff_reshaped.T
    estimatedSignal = estimatedSignal_reshaped.T.reshape(shm_coeff.shape[:-1] + (qb_model_low.B.shape[0],))

    print('shape of estimatedSignal:', estimatedSignal.shape)
    print('shape of shm_coeff:', shm_coeff.shape)
    print('shape of qb_model_low.B:', qb_model_low.B.shape)

    estimatedSignal_scaled = estimatedSignal * b0[..., np.newaxis]
    b0_reshaped = b0[..., np.newaxis]
    resulting_data = np.concatenate((b0_reshaped, estimatedSignal_scaled), axis=-1)
    hdr['dim'][3] = resulting_data.shape[3]
    hdr['descrip'] = 'estimated signal from angularly downsampled data - SCK'

    print('resulting_data:', resulting_data.shape)
    save_nifti(outPrefix + '.nii.gz', resulting_data, affine, hdr)

    max_bval = np.max(bvals)
    new_bvals = np.concatenate(([0], [max_bval] * estimatedSignal.shape[-1]))
    np.savetxt(outPrefix + '.bval', new_bvals, fmt='%d', newline='\n')

    bvals_low, bvecs_low = read_bvals_bvecs(inPrefixLowerAngular + '.bval', inPrefixLowerAngular + '.bvec')
    gtab_low = gradient_table(bvals_low, bvecs_low, b0_threshold=B0_THRESH)
    bvecs_low_filtered = bvecs_low[~gtab_low.b0s_mask, :]

    new_bvecs = np.vstack(([0, 0, 0], bvecs_low_filtered))
    np.savetxt(outPrefix + '.bvec', new_bvecs, fmt='%.8f', newline='\n')

    print('new_bvals:', new_bvals)
    print('new_bvecs:', new_bvecs)
    print('shape of new_bvals:', new_bvals.shape)
    print('shape of new_bvecs:', new_bvecs.shape)

    print('New nifti file saved to', outPrefix + '.nii.gz')
    print('New bvals file saved to', outPrefix + '.bval')
    print('New bvecs file saved to', outPrefix + '.bvec')

resampleDWIgradients(
    imgPath="./testdata/HighAngularResData.nii.gz",
    maskPath="./testdata/mask.nii.gz",
    outPrefix="./testdata/ResampledData",
    imgLowerAngular="./testdata/LowAngularResData.nii.gz"
)
