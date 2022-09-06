import Tools
import numpy as np
import SimpleITK as sitk
import vtk
import pickle
import os
import argparse

def generateModels(age, sex):
    # import Model data
    with open(os.path.join(inputPath, 'ThicknessModel'), "rb") as fp:
        ThicknessModel = pickle.load(fp)

    with open(os.path.join(inputPath, 'IntensityModel'), "rb") as fp:
        IntensityModel = pickle.load(fp)

    with open(os.path.join(inputPath, 'ShapeModel'), "rb") as fp:
        ShapeModel = pickle.load(fp)

    ## target age and sex, and standard deviations from average principal components
    nStdThickness = np.zeros(ThicknessModel[2].shape[0])
    nStdIntensity = np.zeros(IntensityModel[2].shape[0])
    nStdShape = np.zeros(ShapeModel[2].shape[0])
    ## + 2 std away for the second component
    nStdThickness[1] = 2
    nStdIntensity[1] = 2
    nStdShape[1] = 2

    ## Read mask image and average bone segmentation image
    MaskImage = sitk.ReadImage(os.path.join(inputPath, 'SphericalMaskImage.mha'))
    AverageSegmentationImage = sitk.ReadImage(os.path.join(inputPath, 'averageBoneSegmentationSphericalImage.mha'))

    ## Constructing average spherical maps with standard deviation 1 for shape, thickness and intensity

    CoordinateMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(ShapeModel, age, sex, MaskImage=MaskImage,
                                                                      nStd=nStdShape, Coordinates=True)
    CoordinateMap.CopyInformation(AverageSegmentationImage)
    ThicknessMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(ThicknessModel, age, sex, MaskImage=MaskImage,
                                                                     nStd=nStdThickness, Coordinates=False)
    IntensityMap = Tools.ConstrucPredictionSphericalMapsFromPCAModel(IntensityModel, age, sex, MaskImage=MaskImage,
                                                                     nStd=nStdIntensity, Coordinates=False)

    ## Construct external cranial surface mesh with thickness and intensity information
    referenceImage = MaskImage
    referenceImage.CopyInformation(AverageSegmentationImage)
    ExternalSurface = Tools.ConstructCranialSurfaceMeshFromSphericalMaps(CoordinateMap, referenceImage=referenceImage,
                                                                         intensityImageDict={'Density': IntensityMap,
                                                                                             'Thickness': ThicknessMap,
                                                                                             'BoneLabel': AverageSegmentationImage},
                                                                         subsamplingFactor=1, verbose=True)

    ## Create internal cranial surface mesh with external surface and thickness map
    InternalSurface = Tools.CreateInternalSurfaceFromExternalSurface(MaskImage, ExternalSurface=ExternalSurface)

    if sex == 1:
        sex_name = '_male_'
    else:
        sex_name = '_female_'

    ## save the meshes
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(ExternalSurface)
    external_surface_file = 'ExternalSurface' + sex_name + '_Aged' + str(age) + 'years.vtp'
    writer.SetFileName(os.path.join(inputPath, external_surface_file))
    writer.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(InternalSurface)
    internal_surface_file = 'InternalSurface' + sex_name + '_Aged' + str(age) + 'years.vtp'
    writer.SetFileName(os.path.join(inputPath, internal_surface_file))
    writer.Update()


if __name__ == '__main__':
    inputPath = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default=None, help='age to be predicted or starting age')
    parser.add_argument('--sex', default=None, help='sex for prediction, 0 for male, 1 for female, use 2 for both')
    parser.add_argument('--range', default=None, help ='age range for predictions, set --age argument to lower bound')
    parser.add_argument('--interval', default=1, help = 'for 0.1 increments, the increment for age based predictions, when used with range')
    args = parser.parse_args()

    start = float(args.age)
    sex = args.sex

    if args.range is not None:
        end = float(args.range)
        interval = float(args.interval)

        while start <= end:
            if sex == '0':
                generateModels(start, 0)
                start += interval

            elif sex == '1':
                generateModels(start, 1)
                start += interval

            else:
                generateModels(start, 0)
                generateModels(start, 1)
                start += interval

    else:
        if sex == '0':
            generateModels(start, 0)
        elif sex == '1':
            generateModels(start, 1)
        else:
            generateModels(start, 0)
            generateModels(start, 1)