import Toolkit
import vtk
import SimpleITK as sitk
import os
import argparse

def GenerateSphericalMaps(patient_image, segmentation, landmarks):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(landmarks)
    reader.Update()
    spherical_landmarks = reader.GetOutput()

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName('templateGlabellaLandmarks.vtk')
    reader.Update()
    templateLandmarks = reader.GetOutput()

    ## transform
    subjectToTemplateTransform = Toolkit.AlignLandmarksWithTemplate(spherical_landmarks, templateLandmarks, scaling=False)

    ## creating the spherical maps
    binaryBoneImage = Toolkit.CreateBoneMask(patient_image, minimumThreshold=160, maximumThreshold=160)
    boneMesh = Toolkit.CreateMeshFromBinaryImage(binaryBoneImage, insidePixelValue=1)
    externalSurface = Toolkit.CreateContinuousModelOfExternalCranialSurface(boneMesh, spherical_landmarks, maskResult=False,
                                                                            templateLandmarks=templateLandmarks)
    internalSurface = Toolkit.CreateContinuousModelOfInternalCranialSurface(boneMesh, externalSurface, spherical_landmarks)
    cranialBoneSurface = Toolkit.ProjectBoneIntensityOnExternalMesh(patient_image, externalSurface, internalSurface,
                                                                    verbose=True)
    cranialBoneSurface = Toolkit.ProjectImageScalarsOnMesh(cranialBoneSurface, segmentation, label='boneLabel')

    skullSphericalMapSurface = Toolkit.CreateSphericalMapFromSurfaceModel(cranialBoneSurface,
                                                                          subjectToTemplateTransform,
                                                                          landmarks=templateLandmarks,
                                                                          numberOfThetas=200, maskResult=False)

    skullCTIntensitySphericalImage = Toolkit.CreateImageFromSphericalMapModel(skullSphericalMapSurface)
    skullThicknessSphericalImage = Toolkit.CreateImageFromSphericalMapModel(skullSphericalMapSurface,
                                                                            arrayName='Thickness')
    skullEuclideanCoordinatesSphericalImage = Toolkit.CreateVectorImageFromBullsEyeMesh(skullSphericalMapSurface,
                                                                                        arrayName='coords')
    skullBoneSegmentationSphericalImage = Toolkit.CreateImageFromSphericalMapModel(skullSphericalMapSurface,
                                                                                   arrayName='boneLabel')

    ## register with demons algorithm
    averageBoneSegmentation = sitk.ReadImage('averageBoneSegmentationSphericalImage.mha')
    maskImage = sitk.ReadImage('SphericalMaskImage.mha')

    ## the aligned spherical maps can be used in the normaltive model
    transform = Toolkit.RegisterSphericalMapImages(averageBoneSegmentation, skullBoneSegmentationSphericalImage,
                                                   maskImage, initial_transform=None,
                                                   varianceForUpdateField=10, varianceForTotalField=10,
                                                   fitSigmoid=False)

    # Applying the transformation
    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(skullCTIntensitySphericalImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    skullCTIntensitySphericalImage = resampler.Execute(skullCTIntensitySphericalImage)

    resampler.SetReferenceImage(skullThicknessSphericalImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    skullThicknessSphericalImage = resampler.Execute(skullThicknessSphericalImage)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(skullEuclideanCoordinatesSphericalImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    skullEuclideanCoordinatesSphericalImage = resampler.Execute(skullEuclideanCoordinatesSphericalImage)

    return skullCTIntensitySphericalImage, skullThicknessSphericalImage, skullEuclideanCoordinatesSphericalImage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagePath', help='path to patient_specific folder')

    args = parser.parse_args()
    patient_path = '..\\' +  args.imagePath + '\\'

    ctImage = sitk.ReadImage(patient_path + '109A_Tilt_1_noBed.mha')
    segmentationImage = sitk.ReadImage(patient_path + '109ASegmentedSubject.nrrd')
    landmark_mesh = patient_path + '109ALandmarksTemplate.vtp'

    Intensity, Thickness, Coord = GenerateSphericalMaps(ctImage, segmentationImage, landmark_mesh)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(patient_path + 'IntensityMap.mha')
    writer.Execute(Intensity)
    writer.SetFileName(patient_path + 'ThicknessMap.mha')
    writer.Execute(Thickness)
    writer.SetFileName(patient_path + 'CoordMap.mha')
    writer.Execute(Coord)

