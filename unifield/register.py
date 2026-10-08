"""Optional rigid+affine MI registration; original paper settings were not supplied."""
import argparse
from pathlib import Path
import SimpleITK as sitk

def register_pair(lq,hq,iterations=200,seed=42):
    fixed=sitk.Cast(hq,sitk.sitkFloat32);moving=sitk.Cast(lq,sitk.sitkFloat32)
    if fixed.GetDimension()!=3 or moving.GetDimension()!=3:raise ValueError('Registration requires 3D volumes')
    rigid=sitk.CenteredTransformInitializer(fixed,moving,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.GEOMETRY)
    def optimize(transform):
        method=sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(50)
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(.1,seed)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetOptimizerAsGradientDescent(learningRate=1.,numberOfIterations=iterations,convergenceMinimumValue=1e-6,convergenceWindowSize=10)
        method.SetOptimizerScalesFromPhysicalShift()
        method.SetShrinkFactorsPerLevel([4,2,1]);method.SetSmoothingSigmasPerLevel([2,1,0])
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(transform,inPlace=True);method.Execute(fixed,moving)
        return transform
    rigid=optimize(rigid)
    affine=sitk.AffineTransform(3);affine.SetCenter(rigid.GetCenter());affine.SetMatrix(rigid.GetMatrix());affine.SetTranslation(rigid.GetTranslation())
    affine=optimize(affine)
    registered=sitk.Resample(moving,fixed,affine,sitk.sitkLinear,0.,sitk.sitkFloat32)
    return registered,fixed,affine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lq',required=True);p.add_argument('--hq',required=True);p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=200);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    registered,fixed,transform=register_pair(sitk.ReadImage(a.lq),sitk.ReadImage(a.hq),a.iterations,a.seed)
    sitk.WriteImage(registered,str(out/'lq_registered.nii.gz'));sitk.WriteImage(fixed,str(out/'hq.nii.gz'))
    sitk.WriteTransform(transform,str(out/'lq_to_hq.tfm'))
    print('Saved registered pair on HQ grid:',out)
if __name__=='__main__':main()
