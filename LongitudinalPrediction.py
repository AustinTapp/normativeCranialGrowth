
import Tools
import pickle
import SimpleITK as sitk
import vtk
import numpy as np

def LongitudinalPredictionFromPCAModel(MaskImage, functional, Model, originMap, gender, age1, age2, Coordinates = False, additive = True):
    """
        Generate longitudinal spherical maps predictions from PCA regression model outputs
    Parameters
    ----------
    MaskImage: sitk.Image
        mask image
    functional: function
        Regresion function
    Model: list
        PCA model
    originMap: sitk.Image
        original map of the patient
    gender: int
        Target gender (male:1, female:0)
    age1: float
        patient's age at the time of the scan
    age2: float
        target age for prediction
    Coordinates: bool
        Generate coordinate maps or not
    additive: bool
        prediction based on additive or proportional approach

    Returns
    -------
    sitk.Image
        Shape of 500x500 or 500x500x3
    """
    
    Mask = sitk.GetArrayFromImage(MaskImage)
    Masked = np.where(Mask == 1)
    masked = np.append(Masked[0],Masked[1])
    masked = np.reshape(masked,(-1,2),order='F')
    
    mapArray = sitk.GetArrayFromImage(originMap)
    if Coordinates:
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
            PCAParams[i] = ProjectedMap[i] + functional(age2, gender, *Model[2][i,:])[0] - functional(age1, gender, *Model[2][i,:])[0]
        else:
            PCAParams[i] = ((ProjectedMap[i] - functional(age1, gender, *Model[2][i,:])[0]) / Model[3][i].predict(np.array([age1, gender]).reshape(1,-1)) *
                    Model[3][i].predict(np.array([age2, gender]).reshape(1,-1)) +functional(age2, gender, *Model[2][i,:])[0] )

    vals = Model[0] + np.dot(PCAParams, Model[1])

    if not Coordinates:
        MapPred[Mask==1] = vals
        MapPred = sitk.GetImageFromArray(MapPred)
    else:
        MapPred[Mask==1] = vals.reshape((Mask.sum(),3))
        MapPred = sitk.GetImageFromArray(MapPred, isVector=True)

    return(MapPred)

## read aligned spherical maps of the patient for prediction
## please replace the path for the patients of interest
coordinateMap = sitk.ReadImage(
    './alignedEuclideanCoordinatesSphericalImage.mha'
    )
intensityMap = sitk.ReadImage(
    './alignedCTIntensitySphericalImage.mha'
    )
thicknessMap = sitk.ReadImage(
    './alignedThicknessSphericalImage.mha'
    )

## read in mask image and average segmentation map
MaskImage = sitk.ReadImage('./SphericalMaskImage.mha')
print(MaskImage)
AverageSegmentationImage = sitk.ReadImage('./averageBoneSegmentationSphericalImage.mha')

with open('./ThicknessModel', "rb") as fp:
    ThicknessModel = pickle.load(fp)

with open('./IntensityModel', "rb") as fp:
    IntensityModel = pickle.load(fp)

with open('./ShapeModel', "rb") as fp:
    ShapeModel = pickle.load(fp)

## predition

age = 1 ## age in years, at the time of the scan
targetAge = 9 ## target age for prediction, in years
sex = 0 # 1 for male, 0 for female , of the patient

coordPred = LongitudinalPredictionFromPCAModel(
    MaskImage, Tools.Arcsinh, ShapeModel, coordinateMap, gender=sex, age1 = age, age2 = targetAge, Coordinates=True, additive=False
)
coordPred.CopyInformation(AverageSegmentationImage)

thicknessPred = LongitudinalPredictionFromPCAModel(
    MaskImage, Tools.Arcsinh, ThicknessModel, thicknessMap, gender=sex, age1 = age, age2 = targetAge, Coordinates=False,additive=False
)

intensityPred = LongitudinalPredictionFromPCAModel(
    MaskImage, Tools.Arcsinh, IntensityModel, intensityMap, gender=sex, age1 = age, age2 = targetAge, Coordinates=False,additive=False
)


## Construct external cranial surface mesh with thickness and intensity information
referneceImage = MaskImage
referneceImage.CopyInformation(AverageSegmentationImage)
ExternalSurface = Tools.ConstructCranialSurfaceMeshFromSphericalMaps(coordPred, referenceImage=referneceImage,
    intensityImageDict={'Density':intensityPred, 'Thickness': thicknessPred, 'BoneLabel': AverageSegmentationImage}, subsamplingFactor=1,verbose=True)

## Create internal cranial surface mesh with external surface and thickness map
InternalSurface = Tools.CreateInternalSurfaceFromExternalSurface(MaskImage, ExternalSurface=ExternalSurface)

## save the meshes
writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputData(ExternalSurface)
writer.SetFileName('./externalSurface.vtp')
writer.Update()
