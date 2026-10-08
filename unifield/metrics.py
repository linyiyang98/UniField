"""Metrics on normalized MRI intensities; conventions are explicit."""
import numpy as np
from skimage.metrics import structural_similarity

def volume_metrics(prediction,target):
    p=np.clip(np.asarray(prediction,dtype=np.float32),0,1)
    t=np.clip(np.asarray(target,dtype=np.float32),0,1)
    if p.shape!=t.shape or p.ndim!=3:raise ValueError('Expected matching H,W,D volumes')
    mse=float(np.mean((p-t)**2,dtype=np.float64))
    # Finite JSON representation for perfect reconstruction.
    psnr=float(-10*np.log10(max(mse,1e-12)))
    ssim=float(np.mean([structural_similarity(t[:,:,i],p[:,:,i],data_range=1.,win_size=7) for i in range(p.shape[2])]))
    return {'psnr_db':psnr,'ssim_percent':ssim*100,'nrmse_percent':np.sqrt(mse)*100}
