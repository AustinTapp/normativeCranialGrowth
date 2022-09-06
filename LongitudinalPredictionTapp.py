import Tools
import pickle
import SimpleITK as sitk
import vtk
import numpy as np
import os
import argparse

def LongitudinalPredictionFromPCAModel(MaskImage, func, Model, originMap, sex, age1, age2, additive, coords):
    Mask = sitk.GetArrayFromImage(MaskImage)
    Masked = np.where(Mask == 1)
    masked = np.append(Masked[0],Masked[1])
    masked = np.reshape(masked,(-1,2),order='F')
    
    mapArray = sitk.GetArrayFromImage(originMap)
    if coords:
        MapPred = np.zeros((500,500,3))
        MaskedMap = mapArray[Mask==1,:].ravel()
    else:
        MapPred = np.zeros((500,500))
        MaskedMap = mapArray[Mask==1].ravel()
    
    ## projecting to the PCA space
    ProjectedMap = MaskedMap - Model[0]
    ProjectedMap = np.dot(ProjectedMap, Model[1].T)

    PCAParams = np.zeros(Model[2].shape[0])
    # PCAParams2 = np.zeros(GrowthModels[2].shape[0])
    for i in range(Model[2].shape[0]):
        if additive:
            PCAParams[i] = ProjectedMap[i] + func(age2, sex, *Model[2][i,:])[0] - func(age1, sex, *Model[2][i,:])[0]
        else:
            PCAParams[i] = ((ProjectedMap[i] - func(age1, sex, *Model[2][i,:])[0]) / Model[3][i].predict(np.array([age1, sex]).reshape(1,-1)) *
                    Model[3][i].predict(np.array([age2, sex]).reshape(1,-1)) +func(age2, sex, *Model[2][i,:])[0] )

    vals = Model[0] + np.dot(PCAParams, Model[1])

    if not coords:
        MapPred[Mask==1] = vals
        MapPred = sitk.GetImageFromArray(MapPred)
    else:
        MapPred[Mask==1] = vals.reshape((Mask.sum(),3))
        MapPred = sitk.GetImageFromArray(MapPred, isVector=True)

    return MapPred


def prediction(MaskImage, AverageSegmentationImage, coordPred, thicknessPred, intensityPred, age_end, sex, patient_path):
    # Construct external cranial surface mesh with thickness and intensity information
    referenceImage = MaskImage
    referenceImage.CopyInformation(AverageSegmentationImage)
    ExternalSurface = Tools.ConstructCranialSurfaceMeshFromSphericalMaps(coordPred, referenceImage=referenceImage,
                                                                         intensityImageDict={'Density': intensityPred,
                                                                                             'Thickness': thicknessPred,
                                                                                             'BoneLabel': AverageSegmentationImage},
                                                                         subsamplingFactor=1, verbose=True)
    InternalSurface = Tools.CreateInternalSurfaceFromExternalSurface(MaskImage, ExternalSurface=ExternalSurface)

    ## save the meshes
    if sex == 1:
        sex_name = '_male_'
    else:
        sex_name = '_female_'

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(ExternalSurface)
    external_surface_file = 'ExternalSurface' + sex_name + '_Aged' + str(age_end) + 'years.vtp'
    writer.SetFileName(os.path.join(patient_path, external_surface_file))
    writer.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(InternalSurface)
    internal_surface_file = 'InternalSurface' + sex_name + '_Aged' + str(age_end) + 'years.vtp'
    writer.SetFileName(os.path.join(patient_path, internal_surface_file))
    writer.Update()
    return 0


if __name__ == '__main__':
    basePath = os.getcwd()
    patient_specific_path_base = os.path.join(basePath, 'patientImage.mha')

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagePath', default=patient_specific_path_base, help='path to patient_specific folder')
    parser.add_argument('--functional', default=Tools.Arcsinh, help='regression function used in prediction')
    parser.add_argument('--sex', default=None, help='target sex; male = 1, female = 0')
    parser.add_argument('--age_start', default=None, help='patient age at time of scan')
    parser.add_argument('--age_end', default=None, help='target patient age for prediction')
    parser.add_argument('--additive', default=False, help='predict w/ additive (True) or proportional (False) approach')
    args = parser.parse_args()

    functional = args.functional
    CT = args.imagePath + '/' + 'CTTR.mha'
    set_sex = int(args.sex)
    start_age = float(args.age_start)
    end_age = float(args.age_end)
    additive = args.age_start

    # read aligned spherical maps of the patient of interest for prediction
    coordinateMap = sitk.ReadImage(args.imagePath + '/' +'CoordMap.mha')
    intensityMap = sitk.ReadImage(args.imagePath + '/' +'IntensityMap.mha')
    thicknessMap = sitk.ReadImage(args.imagePath + '/' +'ThicknessMap.mha')

    # read in mask image and average segmentation map
    with open('ThicknessModel', "rb") as fp:
        ThicknessModel = pickle.load(fp)
    with open('IntensityModel', "rb") as fp:
        IntensityModel = pickle.load(fp)
    with open('ShapeModel', "rb") as fp:
        ShapeModel = pickle.load(fp)

    mask_image = sitk.ReadImage('SphericalMaskImage.mha')
    AverageSegImage = sitk.ReadImage('averageBoneSegmentationSphericalImage.mha')
    patient_image = sitk.ReadImage(CT)

    coordPred = LongitudinalPredictionFromPCAModel(
        mask_image, functional, ShapeModel, coordinateMap, set_sex, start_age, end_age, additive, coords=True)
    coordPred.CopyInformation(AverageSegImage)

    thicknessPred = LongitudinalPredictionFromPCAModel(
        mask_image, functional, ThicknessModel, thicknessMap, set_sex, start_age, end_age, additive, coords=False)

    intensityPred = LongitudinalPredictionFromPCAModel(
        mask_image, functional, IntensityModel, intensityMap, set_sex, start_age, end_age, additive, coords=False)

    prediction(mask_image, AverageSegImage, coordPred, thicknessPred, intensityPred, end_age, set_sex, args.imagePath)
