import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import skimage
import skimage.measure

def CreateHeadMask(ctImage, hounsfieldThreshold = -200):
    """
    Returns a binary image mask of the head from an input CT image

    Parameters
    ----------
    ctImage: sitkImage
        A scalar CT image
    hounsfieldThreshold: int
        Hard threshold used to binarize the CT image

    Returns
    -------
    sitkImage
        A binary image of the head
    """

    headMask = sitk.GetArrayFromImage(ctImage)

    # Getting the head
    headMask = (headMask > hounsfieldThreshold).astype(np.uint8)

    headMask = skimage.measure.label(headMask)
    largestLabel = np.argmax(np.bincount(headMask.flat)[1:])+1
    headMask = (headMask == largestLabel).astype(np.uint8)

    headMask = sitk.GetImageFromArray(headMask)
    headMask.SetOrigin(ctImage.GetOrigin())
    headMask.SetSpacing(ctImage.GetSpacing())
    headMask.SetDirection(ctImage.GetDirection())

    return headMask

def CreateBoneMask(ctImage, headMaskImage=None, minimumThreshold=160, maximumThreshold=160, verbose=False, ):
    """
    Uses adapting thresholding to create a binary mask of the cranial bones from an input CT image.
    [Dangi et al., Robust head CT image registration pipeline for craniosynostosis skull correction surgery, Healthcare Technology Letters, 2017]

    Parameters
    ----------
    ctImage: sitkImage
        A scalar CT image
    headMaskImage: sitkImage
        A binary image of the head
    minimumThreshold: int
        The lower threshold of the range to use for adapting thresholding
    maximumThreshold: int
        The upper threshold of the range to use for adapting thresholding
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    sitkImage
        A binary image of the cranial bones
    """

    # If a head mask is not provided
    if headMaskImage is None:

        if verbose:
            print('Creating head mask.')

        headMaskImage = CreateHeadMask(ctImage)


    ctImageArray = sitk.GetArrayFromImage(ctImage)
    headMaskImageArray = sitk.GetArrayViewFromImage(headMaskImage)

    # Appling the mask to the CT image
    ctImageArray[headMaskImageArray == 0] = 0

    # Extracting the bones
    minObjects = np.inf
    optimalThreshold = 0
    for threshold in range(minimumThreshold, maximumThreshold+1, 10):

        if verbose:
            print('Optimizing skull segmentation. Threshold {:03d}.'.format(threshold), end='\r')

        labels = skimage.measure.label(ctImageArray >= threshold)
        nObjects = np.max(labels)

        if nObjects < minObjects:
            minObjects = nObjects
            optimalThreshold = threshold
    if verbose:
        print('The optimal threshold for skull segmentation is {:03d}.'.format(optimalThreshold))
    
    ctImageArray = ctImageArray >= optimalThreshold

    ctImageArray = skimage.measure.label(ctImageArray)
    largestLabel = np.argmax(np.bincount(ctImageArray.flat)[1:])+1
    ctImageArray = (ctImageArray == largestLabel).astype(np.uint)
    
    ctImageArray = sitk.GetImageFromArray(ctImageArray)
    ctImageArray.SetOrigin(ctImage.GetOrigin())
    ctImageArray.SetSpacing(ctImage.GetSpacing())
    ctImageArray.SetDirection(ctImage.GetDirection())

    return ctImageArray

def CreateMeshFromBinaryImage(binaryImage, insidePixelValue=1):
    """
    Uses the marching cubes algorithm to create a surface model from a binary image

    Parameters
    ----------
    binaryImage: sitkImage
        The binary image
    insidePixelValue: {int, float}
        The pixel value to use for mesh creation

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """

    numpyImage = sitk.GetArrayViewFromImage(binaryImage).astype(np.ubyte)
    
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(binaryImage.GetSpacing()[0], binaryImage.GetSpacing()[1], binaryImage.GetSpacing()[2])
    vtkImage.SetOrigin(binaryImage.GetOrigin()[0], binaryImage.GetOrigin()[1], binaryImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)

    filter = vtk.vtkMarchingCubes()
    filter.SetInputData(vtkImage)
    filter.SetValue(0, insidePixelValue)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    return mesh

def CreateContinuousModelOfExternalCranialSurface(inputMesh, landmarks, templateLandmarks, numberOfThetas=150, maskResult=True, verbose=False, reversePlane=False):
    """
    Creates a continuous model of the external cranial surface from the bone model segmented from the CT image.
    Landmarking is done via raycasting in spherical coordinates 

    Parameters
    ----------
    inputMesh: vtkPolyData
        Bone surface model
    landmarks: vtkPolyData
        Cranial base landmarks
    numberOfThetas: int
        Sampling resolution in the elevation angle
    maskResult: bool
        Indicates if the resulting mesh will be cropped at the cranial base using the input landmarks
    useGlabella: bool
        if True, the landmarks are located at the glabella, temporal processes of the dorsum sellae and opisthion.
        If False, the landmarks are located at the nasion, temporal processes of the dorsum sellae and opisthion.
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    vtkPolyData
        The resulting external cranial surface model
    """

    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(inputMesh)
    inputMesh = a

    a = vtk.vtkPolyData()
    a.DeepCopy(landmarks)
    landmarks = a

    a = None

    """
    # Wrapping a sphere around the mesh
    """

    inputMesh = CutMeshWithCranialBaseLandmarks(inputMesh, landmarks, extraSpace=20, useTwoLandmarks=True)

    # Warping a sphere to the outer surface
    center = np.zeros([3], dtype=np.float64)
    for p in range(inputMesh.GetNumberOfPoints()):
        center += np.array(inputMesh.GetPoint(p))
    center /= inputMesh.GetNumberOfPoints()

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(350)
    sphere.SetPhiResolution(60)
    sphere.SetThetaResolution(60)
    sphere.SetCenter(center)
    sphere.Update()
    sphere = sphere.GetOutput()

    wrappedSphere = vtk.vtkSmoothPolyDataFilter()
    wrappedSphere.SetInputData(0, sphere)
    wrappedSphere.SetInputData(1, inputMesh)
    wrappedSphere.Update()
    wrappedSphere = wrappedSphere.GetOutput()

    # Subdividing
    filter = vtk.vtkButterflySubdivisionFilter()
    filter.SetInputData(wrappedSphere)
    filter.SetNumberOfSubdivisions(2)
    filter.Update()
    wrappedSphere = filter.GetOutput()

    # Moving points to the bone so adjustment to the shape is perfect
    for p in range(wrappedSphere.GetNumberOfPoints()):
        coords = wrappedSphere.GetPoint(p)
        closestId = inputMesh.FindPoint(coords[0], coords[1], coords[2])
        coords = np.array(inputMesh.GetPoint(closestId))
        wrappedSphere.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])
    inputMesh = wrappedSphere

    """
    # Aligning with the template (only for rotation), and correcting for rotation inaccuracies of the template
    """

    npLandmarks = np.zeros((landmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    npTemplateLandmarks = np.zeros((templateLandmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    for p in range(landmarks.GetNumberOfPoints()):
        npLandmarks[p,:] = landmarks.GetPoint(p)
        npTemplateLandmarks[p,:] = templateLandmarks.GetPoint(p)
    
    center = np.mean(npLandmarks, axis=0)

    R, t = RegisterPointClouds(npLandmarks, npTemplateLandmarks, scaling=False)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.ravel())
    transform.SetCenter(center)
    transform.SetTranslation(t)

    rotationTransform = sitk.Euler3DTransform()
    rotationTransform.SetIdentity()
    rotationTransform.SetRotation(0, 0, -174 * np.pi / 180.0)
    rotationTransform.SetCenter(center)
    

    for p in range(landmarks.GetNumberOfPoints()):
        coords = rotationTransform.TransformPoint(transform.TransformPoint(landmarks.GetPoint(p)))
        landmarks.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    for p in range(inputMesh.GetNumberOfPoints()):
        coords = rotationTransform.TransformPoint(transform.TransformPoint(inputMesh.GetPoint(p)))
        inputMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    """
    # Cutting planes and ray casting initialization
    """

    # Center for ray casting is the dorsum sellae
    center = (np.array(landmarks.GetPoint(1)) + np.array(landmarks.GetPoint(2))) / 2.0

    # Mesh bounds
    meshBounds = np.zeros([6], dtype=np.float64)
    inputMesh.GetBounds(meshBounds)

    ## New mesh
    sampledMesh = vtk.vtkPolyData()
    sampledMesh.SetPoints(vtk.vtkPoints())
    
    ## Calculating the cutting plane
    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))

    # Normal to plane
    dorsumCoords = (landmarkCoords[1, :] + landmarkCoords[2, :]) / 2.0 # Center of dorsum sellae
    midCoords = (landmarkCoords[0, :] + landmarkCoords[3, :]) / 2.0 # Center of nasion and opisthion

    dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :] # Vector of dorsum sellae
    dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

    p0 = landmarkCoords[0, :] + 10 * dorsumVect # Points parallel to dorsum sellae at the sides of the nasion
    p1 = landmarkCoords[0, :] - 10 * dorsumVect
    p2 = landmarkCoords[3, :] # opisthion

    # Normal to the plane nasion - opisthion
    v0 = p2 - p1
    v1 = p2 - p0
    n = np.cross(v0, v1)
    n = n / np.sqrt(np.sum(n**2))

    plane = vtk.vtkPlane()
    if not reversePlane:
        plane.SetNormal(n)
    else:
        plane.SetNormal(-n)
    plane.SetOrigin(p2)
    

    """
    # Sampling
    """
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(inputMesh)
    obbTree.BuildLocator()

    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    thetaDistance = np.pi / (numberOfThetas - 1.0)

    rayEnd = np.zeros([3], dtype=np.float64)
    radius = np.sqrt( (meshBounds[1] - meshBounds[0])**2 + (meshBounds[3] - meshBounds[2])**2 + (meshBounds[5] - meshBounds[4])**2 )
    intersectedPoint = np.zeros([3], dtype=np.float64)

    ## Adding all arrays with one component
    arrayList = []
    for id in range(inputMesh.GetCellData().GetNumberOfArrays()):

        if inputMesh.GetCellData().GetArray(id).GetNumberOfComponents() == 1:
            newArray = vtk.vtkFloatArray()
            newArray.SetName(inputMesh.GetCellData().GetArray(id).GetName())
            newArray.SetNumberOfComponents(1)
            arrayList += [newArray]

    # Adding the array with reconstruction information
    newArray = vtk.vtkFloatArray()
    newArray.SetName('coords')
    newArray.SetNumberOfComponents(3)
    arrayList += [newArray]

    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell()

    nasion = landmarkCoords[0,:]
    dist = np.linalg.norm(nasion - center) # Distance nasion to center

    nasionSinTheta =  (nasion[2] - center[2])/dist
    nasionSinPhi = (nasion[1] - center[1]) / (dist * np.sqrt( (1 - nasionSinTheta**2) ) )

    opisthion = landmarkCoords[3,:]
    dist = np.linalg.norm(opisthion - center) # Distance nasion to center

    opisthionSinTheta =  (opisthion[2] - center[2])/dist
    opisthionSinPhi = (opisthion[1] - center[1]) / (dist * np.sqrt( (1 - opisthionSinTheta**2) ) )

    for latitude in range(numberOfThetas):

        if verbose:
            print('Sampling spherical space: {:05d}/{:05d}'.format(latitude, numberOfThetas), end='\r')

        theta = np.pi/2 - latitude*np.pi/(numberOfThetas-1)

        thetaLongitude = 2.0*np.pi*np.cos(theta)
        nPointsAtTheta = int(np.floor(thetaLongitude / thetaDistance))

        for longitude in range(nPointsAtTheta):
            if nPointsAtTheta != 1:
                phi = -np.pi + longitude*2*np.pi / (nPointsAtTheta - 1)
            else:
                phi = 0


            distToNasion = (nasionSinPhi-np.sin(phi))**2
            distToOpisthion = (opisthionSinPhi-np.sin(phi))**2
            
            limitTheta = nasionSinTheta * distToOpisthion/(distToNasion + distToOpisthion) + opisthionSinTheta * distToNasion/(distToNasion + distToOpisthion)

            if not maskResult or np.sin(theta) >= limitTheta:
            
                rayEnd[0] = center[0] + radius*np.cos(phi)*np.cos(theta)
                rayEnd[1] = center[1] + radius*np.sin(phi)*np.cos(theta)
                rayEnd[2] = center[2] + radius*np.sin(theta)

                if obbTree.IntersectWithLine(center, rayEnd, intersectionPoints, intersectionCellIds):
                    closestDist = 0
                    closestId = 0
                    for p in range(intersectionPoints.GetNumberOfPoints()):

                        intersectionPoints.GetPoint(p, intersectedPoint)
                        dist = np.sqrt((center[0] - intersectedPoint[0])**2 + (center[1] - intersectedPoint[1])**2 + (center[2] - intersectedPoint[2])**2 )
                    
                        if dist > closestDist:
                            closestDist = dist
                            closestId = p

                    intersectionPoints.GetPoint(closestId, intersectedPoint)
                    # Intersected point has the Euclidean coordinates of the points at the specific longitude and latitude

                    # Finding the color
                    pCoords = np.zeros([3], dtype=np.float64)
                    weights = np.zeros([3], dtype=np.float64)
                    subId = vtk.reference(0)
                    cellId = cellLocator.FindCell(intersectedPoint, 0, triangle, pCoords, weights)

                    #Calculating the cartesian coordinates for the spherical map
                    angle = phi
                    rho = float(latitude)/(numberOfThetas-1)

                    x = rho * np.cos(phi)
                    y = rho * np.sin(phi)


                    if maskResult:
                        toDraw = plane.EvaluateFunction(intersectedPoint) > 0.1
                    else:
                        toDraw = True

                    if toDraw:
                        sampledMesh.GetPoints().InsertNextPoint(x, y, 0)

                        for arrayId in range(len(arrayList)-1):
                    
                            if cellId >= 0:
                                arrayList[arrayId].InsertNextTuple1(inputMesh.GetCellData().GetArray(arrayList[arrayId].GetName()).GetTuple1(cellId))
                            else:
                                arrayList[arrayId].InsertNextTuple1(0)

                        arrayList[-1].InsertNextTuple3(intersectedPoint[0], intersectedPoint[1], intersectedPoint[2])

    for thisArray in arrayList:
        sampledMesh.GetPointData().AddArray(thisArray)

    """
    # Triagulating and recosntructing from spherical to cranial mesh
    """
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(sampledMesh)
    filter.Update()
    sampledMesh = filter.GetOutput()
    if verbose:
        print()

    for p in range(sampledMesh.GetNumberOfPoints()):
        coords = sampledMesh.GetPointData().GetArray('coords').GetTuple3(p)
        sampledMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    sampledMesh.GetPointData().RemoveArray('coords')

    # Cleaning: eliminating duplicated points and cells
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(sampledMesh)
    filter.PointMergingOn()
    filter.Update()
    sampledMesh = filter.GetOutput()

    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(sampledMesh)
    filter.Update()
    sampledMesh = filter.GetOutput()

    """
    # Transforming back to the original coordinates space
    """
    transform = transform.GetInverse()
    rotationTransform = rotationTransform.GetInverse()
    for p in range(sampledMesh.GetNumberOfPoints()):
        coords = transform.TransformPoint(rotationTransform.TransformPoint(sampledMesh.GetPoint(p)))
        sampledMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    """
    # Smoothing to avoid problems with normals when the bone model comes from low quality images
    """
    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(sampledMesh)
    filter.SetNumberOfIterations(30)
    filter.Update()
    sampledMesh = filter.GetOutput()
    

    """
    # Calculating normals
    """
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(sampledMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    sampledMesh = filter.GetOutput()

    return sampledMesh

def CreateContinuousModelOfInternalCranialSurface(boneMesh, externalMesh, landmarks, verbose=False):
    """
    Creates a continuous model of the internal cranial surface from the bone model segmented from the CT image and the external surface.

    Parameters
    ----------
    boneMesh: vtkPolyData
        Bone surface model
    externalMesh: vtkPolyData
        External cranial surface
    landmarks: vtkPolyData
        Cranial base landmarks
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    vtkPolyData
        The resulting internal cranial surface model
    """

    # Copying the external mesh
    internalMesh = vtk.vtkPolyData()
    internalMesh.DeepCopy(externalMesh) 

    # Shrinking following normals
    for p in range(internalMesh.GetNumberOfPoints()):
        
        coords = np.array(internalMesh.GetPoint(p))
        normal = np.array(internalMesh.GetPointData().GetArray('Normals').GetTuple3(p))

        coords -= 20 * normal # Shrinking by 2 cm, which should be more than bone thickness

        internalMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])
    
    
    # Raycasting between homologous points at the internal and external surfaces, and finding internsections with the cut bone surface
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(boneMesh)
    obbTree.BuildLocator()

    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    intersectionFoundArray = vtk.vtkCharArray()
    intersectionFoundArray.SetName('IntersectionFound')
    intersectionFoundArray.SetNumberOfComponents(1)

    for p in range(internalMesh.GetNumberOfPoints()):

        endCoords = np.array(internalMesh.GetPoint(p)) 
        startCoords = np.array(externalMesh.GetPoint(p))  - np.array(externalMesh.GetPointData().GetArray('Normals').GetTuple3(p)) * 0.5

        if obbTree.IntersectWithLine(startCoords, endCoords, intersectionPoints, intersectionCellIds) and intersectionPoints.GetNumberOfPoints() > 0:

            closestDist = np.Inf
            closestId = 0
            for q in range(intersectionPoints.GetNumberOfPoints()):
                    
                intersectedPoint = np.array(intersectionPoints.GetPoint(q))
                dist = np.linalg.norm(intersectedPoint - startCoords)
                    
                if dist < closestDist and dist > 0.3:
                    closestDist = dist
                    closestId = q

            internalMesh.GetPoints().SetPoint(p, intersectionPoints.GetPoint(closestId))

            intersectionFoundArray.InsertNextTuple1(1)
        else:
            intersectionFoundArray.InsertNextTuple1(0)

    
    internalMesh.GetPointData().AddArray(intersectionFoundArray)

    

    for iter in range(500):

        cost = 0

        for p in range(internalMesh.GetNumberOfPoints()):

            # Intesection was not found
            if intersectionFoundArray.GetTuple1(p) == 0:

                cost +=1
                
                #Average the position of the neighbors if their intersection was found
                neighboringPoints = []

                neighboringCellList = vtk.vtkIdList() # Finding neighboring cells
                internalMesh.GetPointCells(p, neighboringCellList)
                for c in range(neighboringCellList.GetNumberOfIds()): # Iterating the neighboring cells of this poin
                    cellPoints = vtk.vtkIdList()
                    internalMesh.GetCellPoints(neighboringCellList.GetId(c), cellPoints)
                    for q in range(cellPoints.GetNumberOfIds()): # Iterating the points in the neighboring cells
                        if cellPoints.GetId(q) != p and cellPoints.GetId(q) not in neighboringPoints and intersectionFoundArray.GetTuple1(cellPoints.GetId(q)) == 1:
                            neighboringPoints += [cellPoints.GetId(q)]
             
                # Averaging neighbors with information
                if len(neighboringPoints) > 0:
                    averageThickness = 0
                    for q in neighboringPoints:
    
                        coords = np.array(internalMesh.GetPoint(q))
                        externalCoords = np.array(externalMesh.GetPoint(q))
                        averageThickness += np.linalg.norm(coords - externalCoords)
                    averageThickness /= len(neighboringPoints)

                    coords = np.array(externalMesh.GetPoint(p)) - np.array(externalMesh.GetPointData().GetArray('Normals').GetTuple3(p)) * averageThickness
                    internalMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])
                    intersectionFoundArray.SetTuple1(p, 1)

        if verbose:
            print('Iteration {:03d}. Cost: {:04.2f}.'.format(iter, cost), end='\r')
        
    internalMesh.GetPointData().RemoveArray('IntersectionFound')

    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(internalMesh)
    filter.SetNumberOfIterations(20)
    filter.Update()
    internalMesh = filter.GetOutput()

    # Warping a sphere to the outer surface
    center = np.zeros([3], dtype=np.float64)
    for p in range(internalMesh.GetNumberOfPoints()):
        center += np.array(internalMesh.GetPoint(p))
    center /= internalMesh.GetNumberOfPoints()

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(350)
    sphere.SetPhiResolution(40)
    sphere.SetThetaResolution(40)
    sphere.SetCenter(center)
    sphere.Update()
    sphere = sphere.GetOutput()

    wrappedSphere = vtk.vtkSmoothPolyDataFilter()
    wrappedSphere.SetInputData(0, sphere)
    wrappedSphere.SetInputData(1, internalMesh)
    wrappedSphere.Update()
    wrappedSphere = wrappedSphere.GetOutput()

    # Subdividing
    filter = vtk.vtkButterflySubdivisionFilter()
    filter.SetInputData(wrappedSphere)
    filter.SetNumberOfSubdivisions(1)
    filter.Update()
    wrappedSphere = filter.GetOutput()

    # Moving points to the bone so adjustment to the shape is perfect
    for p in range(wrappedSphere.GetNumberOfPoints()):
        coords = wrappedSphere.GetPoint(p)
        closestId = internalMesh.FindPoint(coords[0], coords[1], coords[2])
        coords = np.array(internalMesh.GetPoint(closestId))
        wrappedSphere.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])
    inputMesh = wrappedSphere
    

    # Making sure all points are above the cranial base planes
    internalMesh = CutMeshWithCranialBaseLandmarks(internalMesh, landmarks, extraSpace=20, useTwoLandmarks=True)

    # Collecting only the largest connected component (to remove noise at cranial base)
    filter = vtk.vtkPolyDataConnectivityFilter()
    filter.SetInputData(internalMesh)
    filter.SetExtractionModeToLargestRegion()
    filter.Update()
    internalMesh = filter.GetOutput()
    
    # Cleaning: eliminating duplicated points and cells
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(internalMesh)
    filter.PointMergingOn()
    filter.Update()
    internalMesh = filter.GetOutput()

    # Making sure there are only triangles
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(internalMesh)
    filter.Update()
    internalMesh = filter.GetOutput()

    # Smoothing to avoid weird normals
    filter = vtk.vtkSmoothPolyDataFilter()
    filter.SetInputData(internalMesh)
    filter.SetNumberOfIterations(20)
    filter.Update()
    internalMesh = filter.GetOutput()

    # Calculating normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(internalMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    internalMesh = filter.GetOutput()

    return internalMesh

def ProjectBoneIntensityOnExternalMesh(ctImage, externalMesh, internalMesh, verbose=False):
    """
    Samples the space between the external and internal cranial surfaces and projects the average bone intensity in the external cranial surface 

    Parameters
    ----------
    ctImage: sitkImage
        CT image
    externalMesh: vtkPolyData
        External cranial surface
    internalMesh: vtkPolyData
        Internal cranial surface
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    vtkPolyData
        The external cranial surface with the local average CT image intensity
    """

    # Copying the mesh to avoid modifications
    projectedMesh = vtk.vtkPolyData()
    projectedMesh.DeepCopy(externalMesh)

    nDivisions = 3 # divisions between the external and internal layers

    # Creating arrays
    intensityArray = vtk.vtkFloatArray()
    intensityArray.SetName('ImageIntensity')
    intensityArray.SetNumberOfComponents(1)
    intensityArray.SetNumberOfTuples(externalMesh.GetNumberOfCells())
    projectedMesh.GetCellData().AddArray(intensityArray)

    thicknessArray = vtk.vtkFloatArray()
    thicknessArray.SetName('Thickness')
    thicknessArray.SetNumberOfComponents(1)
    thicknessArray.SetNumberOfTuples(externalMesh.GetNumberOfCells())
    projectedMesh.GetCellData().AddArray(thicknessArray)

    # Raycasting between homologous points at the internal and external surfaces, and finding internsections with the cut bone surface
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(internalMesh)
    obbTree.BuildLocator()
    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    for c in range(projectedMesh.GetNumberOfCells()):

        cellIntensity = 0
        cellThickness = 0
        nValidPointsInCell = 0

        pointIds = vtk.vtkIdList()
        projectedMesh.GetCellPoints(c, pointIds)
        for p in range(pointIds.GetNumberOfIds()):

            pointId = pointIds.GetId(p)

            pointThickness = 0
            pointIntensity = 0
            nValidPointDivisions = 0

            externalCoords = np.array(projectedMesh.GetPoint(pointId))
            closestId = internalMesh.FindPoint(externalCoords)

            if closestId >=0:

                internalCoords = np.array(internalMesh.GetPoint(closestId))

                thickness = np.linalg.norm(externalCoords - internalCoords)

                for q in range(nDivisions):

                    coords = externalCoords + (internalCoords - externalCoords) * (q+1)/(nDivisions+1)

                    try:
                        imageIndex = ctImage.TransformPhysicalPointToIndex(coords)
                        intensity = ctImage.GetPixel(imageIndex)
                        
                        pointIntensity += intensity
                        nValidPointDivisions += 1
                    except:
                        pass

                pointThickness = thickness
            
            cellThickness += pointThickness
            
            if nValidPointDivisions > 0:
                pointIntensity /= nValidPointDivisions
                cellIntensity += pointIntensity
                nValidPointsInCell += 1

        if nValidPointsInCell > 0:
            cellThickness /= nValidPointsInCell
            cellIntensity /= nValidPointsInCell

        thicknessArray.SetTuple1(c, cellThickness)
        intensityArray.SetTuple1(c, cellIntensity)

    return projectedMesh

def ProjectImageScalarsOnMesh(mesh, image, label):

    numpyImage = sitk.GetArrayViewFromImage(image).astype(np.ubyte)
    
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[2])
    vtkImage.SetOrigin(image.GetOrigin()[0], image.GetOrigin()[1], image.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)

    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputData(vtkImage)
    thresholdFilter.ThresholdBetween(1, 7)
    thresholdFilter.Update()
    segmentedVolume = thresholdFilter.GetOutput()

    dataArray = vtk.vtkIntArray()
    dataArray.SetName(label)
    dataArray.SetNumberOfComponents(1)
    dataArray.SetNumberOfTuples(mesh.GetNumberOfPoints())

    cellDataArray = vtk.vtkIntArray()
    cellDataArray.SetName(label)
    cellDataArray.SetNumberOfComponents(1)
    cellDataArray.SetNumberOfTuples(mesh.GetNumberOfCells())

    for p in range(mesh.GetNumberOfPoints()):
        
        coords = mesh.GetPoint(p)

        closestId = segmentedVolume.FindPoint(coords)

        dataArray.SetTuple1(p, segmentedVolume.GetPointData().GetScalars().GetTuple1(closestId))

    mesh.GetPointData().AddArray(dataArray)

    coords = np.zeros((3), dtype=np.float32)
    for c in range(mesh.GetNumberOfCells()):

        cell = mesh.GetCell(c)
        cell.TriangleCenter(cell.GetPoints().GetPoint(0), cell.GetPoints().GetPoint(1), cell.GetPoints().GetPoint(2), coords)

        closestId = segmentedVolume.FindPoint(coords)

        cellDataArray.SetTuple1(c, segmentedVolume.GetPointData().GetScalars().GetTuple1(closestId))

    mesh.GetCellData().AddArray(cellDataArray)

    return mesh

def CreateSphericalMapFromSurfaceModel(inputMesh, subjectToTemplateTransform, landmarks, numberOfThetas=100, maskResult=True, verbose=False):
    """
    Creates a spherical map representation of a cranial bone surface model.

    Parameters
    ----------
    inputMesh: vtkPolyData
        Cranial bone surface model
    subjectToTemplateTransform: sitkTransform
        Transformation form the subject to the reference/template space
    numberOfThetas: int
        Sampling resolution in the elevation angle
    maskResult: bool
        Indicates whether the result will be cropped using the cranial base landmarks or not. Only the first and fourth landmark are used for masking.
    useGlabella: bool
        if True, the landmarks are located at the glabella, temporal processes of the dorsum sellae and opisthion.
        If False, the landmarks are located at the nasion, temporal processes of the dorsum sellae and opisthion.
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    vtkPolyData
        The spherical map as a flat surface model
    """

    # Creating a copy of the input models
    a = vtk.vtkPolyData()
    a.DeepCopy(inputMesh)
    inputMesh = a

    # Transforming the cranial bone surface to the template space
    for p in range(inputMesh.GetNumberOfPoints()):
        coords = subjectToTemplateTransform.TransformPoint(inputMesh.GetPoint(p))
        inputMesh.GetPoints().SetPoint(p, coords[0], coords[1], coords[2])

    meshBounds = np.zeros([6], dtype=np.float64)
    inputMesh.GetBounds(meshBounds)
    center = np.zeros((3), dtype=np.float32)
    
    # Calculating the plane to mask out the information below the cranial base landmarks
    # Important: We only cut using the first and fourth landmark!
    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))

    dorsumCoords = (landmarkCoords[1, :] + landmarkCoords[2, :]) / 2.0 # center of dursum
    midCoords = (landmarkCoords[0, :] + landmarkCoords[3, :]) / 2.0 # Center of cranial base

    dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :] # Left to right vector in the dursum sellae
    dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

    p0 = landmarkCoords[0, :] + 10 * dorsumVect # Two points in the forehead
    p1 = landmarkCoords[0, :] - 10 * dorsumVect #
    p2 = landmarkCoords[3, :] # the opisthion

    v0 = p2 - p1
    v1 = p2 - p0
    n = np.cross(v0, v1)
    n = n / np.sqrt(np.sum(n**2))

    plane = vtk.vtkPlane()
    plane.SetNormal(-n)
    plane.SetOrigin(p2)

    # Sampling
    sphericalMesh = vtk.vtkPolyData()
    sphericalMesh.SetPoints(vtk.vtkPoints())

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(inputMesh)
    obbTree.BuildLocator()

    intersectionPoints = vtk.vtkPoints()
    intersectionCellIds = vtk.vtkIdList()

    thetaDistance = np.pi / (numberOfThetas - 1.0)

    rayEnd = np.zeros([3], dtype=np.float64)
    radius = np.sqrt( (meshBounds[1] - meshBounds[0])**2 + (meshBounds[3] - meshBounds[2])**2 + (meshBounds[5] - meshBounds[4])**2 )
    intersectedPoint = np.zeros([3], dtype=np.float64)

    ## Adding all arrays with one component
    arrayList = []
    for id in range(inputMesh.GetCellData().GetNumberOfArrays()):

        if inputMesh.GetCellData().GetArray(id).GetNumberOfComponents() == 1:
            newArray = vtk.vtkFloatArray()
            newArray.SetName(inputMesh.GetCellData().GetArray(id).GetName())
            newArray.SetNumberOfComponents(1)
            arrayList += [newArray]

    # Adding the array with reconstruction information
    newArray = vtk.vtkFloatArray()
    newArray.SetName('coords')
    newArray.SetNumberOfComponents(3)
    arrayList += [newArray]

    
    # This speeds up finding the intersecitons when doing ray-casting
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell() # Allocating memory

    # Spherical coordinates of the first (glabella or nasion) and fourth (opisthion) landmarks
    nasion = np.array(landmarks.GetPoint(0))
    dist = np.sqrt(np.sum(nasion**2))

    nasionSinTheta =  (nasion[2] - center[2])/dist
    nasionSinPhi = (nasion[1] - center[1]) / (dist * np.sqrt( (1 - nasionSinTheta**2) ) )

    opisthion = np.array(landmarks.GetPoint(3))
    dist = np.sqrt(np.sum(opisthion**2))

    opisthionSinTheta =  (opisthion[2] - center[2])/dist
    opisthionSinPhi = (opisthion[1] - center[1]) / (dist * np.sqrt( (1 - opisthionSinTheta**2) ) )

    for latitude in range(numberOfThetas): # elevation sampling

        if verbose:
            print('Creating bulls eye mesh: {:05d}/{:05d}'.format(latitude, numberOfThetas), end='\r')

        theta = np.pi/2 - latitude*np.pi/(numberOfThetas-1) # Elevation angle

        thetaLongitude = 2.0*np.pi*np.cos(theta)
        nPointsAtTheta = int(np.floor(thetaLongitude / thetaDistance))

        for longitude in range(nPointsAtTheta): # Azimuth sampling
            if nPointsAtTheta != 1:
                phi = -np.pi + longitude*2*np.pi / (nPointsAtTheta - 1) # Azimuth angle
            else:
                phi = 0


            distToNasion = (nasionSinPhi-np.sin(phi))**2
            distToOpisthion = (opisthionSinPhi-np.sin(phi))**2
            
            limitTheta = nasionSinTheta * distToOpisthion/(distToNasion + distToOpisthion) + opisthionSinTheta * distToNasion/(distToNasion + distToOpisthion)

            if not maskResult or np.sin(theta) >= limitTheta:
            
                # Ray-casting
                rayEnd[0] = center[0] + radius*np.cos(phi)*np.cos(theta)
                rayEnd[1] = center[1] + radius*np.sin(phi)*np.cos(theta)
                rayEnd[2] = center[2] + radius*np.sin(theta)

                if obbTree.IntersectWithLine(center, rayEnd, intersectionPoints, intersectionCellIds):
                    closestDist = np.inf
                    closestId = 0
                    for p in range(intersectionPoints.GetNumberOfPoints()):

                        intersectionPoints.GetPoint(p, intersectedPoint)
                        dist = np.sqrt((center[0] - intersectedPoint[0])**2 + (center[1] - intersectedPoint[1])**2 + (center[2] - intersectedPoint[2])**2 )
                    
                        if dist < closestDist:
                            closestDist = dist
                            closestId = p

                    intersectionPoints.GetPoint(closestId, intersectedPoint)
                    # Intersected point has the Euclidean coordinates of the points at the specific longitude and latitude

                    # Finding the color
                    pCoords = np.zeros([3], dtype=np.float64)
                    weights = np.zeros([3], dtype=np.float64)
                    subId = vtk.reference(0)
                    cellId = cellLocator.FindCell(intersectedPoint, 0, triangle, pCoords, weights)

                    #Calculating the cartesian coordinates for the BullsEye
                    angle = phi
                    rho = float(latitude)/(numberOfThetas-1)

                    x = rho * np.cos(phi)
                    y = rho * np.sin(phi)


                    if maskResult:
                        toDraw = plane.EvaluateFunction(intersectedPoint) > 0.1
                    else:
                        toDraw = True

                    if toDraw:

                        sphericalMesh.GetPoints().InsertNextPoint(x, y, 0)

                        for arrayId in range(len(arrayList)-1):
                    
                            if cellId >= 0:
                                arrayList[arrayId].InsertNextTuple1(inputMesh.GetCellData().GetArray(arrayList[arrayId].GetName()).GetTuple1(cellId))
                            else:
                                arrayList[arrayId].InsertNextTuple1(0)

                        arrayList[-1].InsertNextTuple3(intersectedPoint[0], intersectedPoint[1], intersectedPoint[2])

    for thisArray in arrayList:
        sphericalMesh.GetPointData().AddArray(thisArray)

    # Triangulating the points
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(sphericalMesh)
    filter.Update()
    sphericalMesh = filter.GetOutput()
    if verbose:
        print()

    return sphericalMesh

def CreateImageFromSphericalMapModel(inputMesh, arrayName='ImageIntensity', imageSize=500, imageExtent=2.0, zeroNegativeValues=True, verbose=False):
    """
    Creates a 2D image of the spherical map model of the cranial bone surface using a scalar array from the model

    Parameters
    ----------
    inputMesh: vtkPolyData
        Spherical map model
    arrayName: string
        name of the array in the model to use to create the image 
    imageSize: int
        Resolution in pixels of the image created
    imageExtent: int
        Extent of the coordinates of the image. The image coordinates are [-imageExtent/2.0, imageExtent/2)
    zeroNegativeValues: bool
         If True, all negative values in the array will be set to zero in the image. 
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    np.array
        An 2D image of the array with arrayName in the spherical map
        The background in the image is set to -1
    """
    
    # Creating the image
    image = sitk.Image(imageSize, imageSize, sitk.sitkFloat32)
    image.SetOrigin([-imageExtent/2.0, -imageExtent/2.0])
    image.SetSpacing([imageExtent/(imageSize-1), imageExtent/(imageSize-1)])

    # The array in the model
    colorArray = inputMesh.GetPointData().GetArray(arrayName)

    # Memory allocation
    closestCoords = np.zeros([3], dtype=np.float32)
    pCoords = np.zeros([3], dtype=np.float32)
    w = np.zeros([3], dtype=np.float32)
    triangle = vtk.vtkGenericCell()

    # Cell locator to speed up cell search
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    for x in range(imageSize):

        if verbose:
            print('Creating bullseye image. Row {:03d}.'.format(x), end='\r')

        for y in range(imageSize):

            xCoords = image.GetOrigin()[0] + image.GetSpacing()[0] * x
            yCoords = image.GetOrigin()[1] + image.GetSpacing()[1] * y

            coords = (xCoords, yCoords, 0)

            cellId = cellLocator.FindCell(coords, 0, triangle, pCoords, w)
            if cellId >= 0:
                pointId = inputMesh.FindPoint(coords)

                if pointId >= 0:

                    if zeroNegativeValues:
                        image.SetPixel((x,y), max(0, colorArray.GetTuple1(pointId)))
                    else:
                        image.SetPixel((x,y), colorArray.GetTuple1(pointId))
                else:
                    image.SetPixel((x,y), -1)
            else:
                image.SetPixel((x,y), -1)

    return image

def CutMeshWithCranialBaseLandmarks(mesh, landmarks, extraSpace=0, useTwoLandmarks=False):
    """
    Crops the input surface model using the planes defined by the input landmarks

    Parameters
    ----------
    mesh: vtkPolyData
        Cranial surface model
    landmarks: vtkPolyData
        Cranial base landmarks (4 points)
    extraSpace: int
        Indicates the amount of extract space to keep under the planes defined by the cranial base landmarks
    useTwoLandmarks: bool
        Indicates if the cut is done only using the first and fourth landmarks, or using the two planes defined by all the landmarks

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """


    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))
    if not useTwoLandmarks:
        

        # normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
    
        ###########
        ## Moving landmark coordinates 1 cm away from cranial base so we don't miss the squamousal suture

        distanceToMove = (extraSpace/100.0) * np.abs(np.dot(np.mean(landmarkCoords[1:3,:], axis=0, keepdims=False) - landmarkCoords[3,:], n0))

        landmarkCoords[1:3,:] +=  (n0*distanceToMove).reshape((1,3))

        # Recalculating normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
        ###########
    
        # normal of second plane
        v0 = landmarkCoords[2, :] - landmarkCoords[3, :] # For plane 2 (posterior)
        v1 = landmarkCoords[1, :] - landmarkCoords[3, :]
        n1 = np.cross(v0, v1)
        n1 = n1 / np.sqrt(np.sum(n1**2))

        plane1 = vtk.vtkPlane()
        plane1.SetNormal(-n0)
        plane2 = vtk.vtkPlane()
        plane2.SetNormal(-n1)

        plane1.SetOrigin(landmarkCoords[0,:])
        plane2.SetOrigin(landmarkCoords[3,:])

        intersectionFunction = vtk.vtkImplicitBoolean()
        intersectionFunction.AddFunction(plane1)
        intersectionFunction.AddFunction(plane2)
        intersectionFunction.SetOperationTypeToIntersection()
    else:
        
        if extraSpace > 0:
            # normal of first plane
            v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
            v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[0,:] +=  (n0*extraSpace)

            # normal of second plane
            v0 = landmarkCoords[3, :] - landmarkCoords[2, :] # For plane 1 (frontal)
            v1 = landmarkCoords[3, :] - landmarkCoords[1, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[3,:] +=  (n0*extraSpace)
        
        # Normal to plane
        dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :]
        dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

        p0 = landmarkCoords[0, :] + 10 * dorsumVect
        p1 = landmarkCoords[0, :] - 10 * dorsumVect
        p2 = landmarkCoords[3, :]


        v0 = p2 - p1
        v1 = p2 - p0
        n = np.cross(v0, v1)
        n = n / np.sqrt(np.sum(n**2))


        plane = vtk.vtkPlane()
        plane.SetNormal(-n)
        plane.SetOrigin(p2)


        intersectionFunction = plane

    #cutter = vtk.vtkClipPolyData()
    cutter = vtk.vtkExtractPolyDataGeometry()
    cutter.ExtractInsideOff()
    cutter.SetInputData(mesh)
    #cutter.SetClipFunction(intersectionFunction)
    cutter.SetImplicitFunction(intersectionFunction)
    cutter.Update()

    return cutter.GetOutput()

def RegisterPointClouds(A, B, scaling=False):
    """
    Calculates analytically the least-squares best-fit transform between corresponding 3D points A->B.

    Parameters
    ----------
    A: np.array
        Moving point cloud with shape Nx3, where N is the number of points
    B: np.array
        Fixed point cloud with shape Nx3, where N is the number of points
    scaling: bool
        Indicates if the calculated transformation is purely rigid (False) or contains isotropic scaling (True)

    Returns
    -------
    np.array
        Rotation (+scaling) matrix with shape 3x3
    np.array
        Translation vector with shape 3
    """

    assert len(A) == len(B) # Both point clouds must have the same number of points
    
    zz = np.zeros(shape=[A.shape[0],1])
    A = np.append(A, zz, axis=1)
    B = np.append(B, zz, axis=1)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Scaling
    if scaling:
        AA = np.dot(R.T, AA.T).T
        s = np.mean(np.linalg.norm(BB[:,:3], axis=1)) / np.mean(np.linalg.norm(AA[:,:3], axis=1))
        R *= s

    R = R[:3,:3]#.T
    t = (centroid_B - centroid_A)[:3]

    return R, t

def CreateVectorImageFromBullsEyeMesh(inputMesh, arrayName='coords', imageSize=500, imageExtent=2.0, verbose=False):
    """
    Creates a 2D image of the spherical map model of the cranial bone surface, using a vector array from the model

    Parameters
    ----------
    inputMesh: vtkPolyData
        Spherical map model
    arrayName: string
        name of the array in the model to use to create the image 
    imageSize: int
        Resolution in pixels of the image created
    imageExtent: int
        Extent of the coordinates of the image. The image coordinates are [-imageExtent/2.0, imageExtent/2)
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    np.array
        An 2D image of the array with arrayName in the spherical map
        The background in the image is set to -1
    """

    colorArray = inputMesh.GetPointData().GetArray(arrayName)
    nComponents = colorArray.GetNumberOfComponents()
    
    image = sitk.Image([imageSize, imageSize], sitk.sitkVectorFloat32, nComponents)
    image.SetOrigin([-imageExtent/2.0, -imageExtent/2.0])
    image.SetSpacing([imageExtent/(imageSize-1), imageExtent/(imageSize-1)])

    # Setting -1 for areas without information
    zeroCoords = [-1] * nComponents    

    closestCoords = np.zeros([3], dtype=np.float32)
    pCoords = np.zeros([3], dtype=np.float32)
    w = np.zeros([3], dtype=np.float32)

    # Cell locator
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(inputMesh) 
    cellLocator.BuildLocator()

    triangle = vtk.vtkGenericCell()

    for x in range(imageSize):
        if verbose:
            print('Creating bullseye image. Row {:03d}.'.format(x), end='\r')

        for y in range(imageSize):

            xCoords = image.GetOrigin()[0] + image.GetSpacing()[0] * x
            yCoords = image.GetOrigin()[1] + image.GetSpacing()[1] * y

            coords = (xCoords, yCoords, 0)

            #cellId = inputMesh.FindCell(coords, None, 0, 0.1, subId, pCoords, w)
            cellId = cellLocator.FindCell(coords, 0, triangle, pCoords, w)
            if cellId >= 0:
                pointId = inputMesh.FindPoint(coords)
                if pointId >= 0:
                    image[x,y] = colorArray.GetTuple(pointId)
                else:
                    image[x,y] = zeroCoords
            else:
                image[x,y] = zeroCoords

    return image

def AlignLandmarksWithTemplate(landmarks, templateLandmarks, scaling=False, verbose=False):

    """
    Calculates analytically the least-squares best-fit transform between corresponding 3D points A->B.

    Parameters
    ----------
    landmarks: vtkPolyData
        Cranial base landmarks
    scaling: bool
        Indicates if the calculated transformation is purely rigid (False) or contains scaling (True)
    useGlabella: bool
        if True, the landmarks are located at the glabella, temporal processes of the dorsum sellae and opisthion.
        If False, the landmarks are located at the nasion, temporal processes of the dorsum sellae and opisthion.
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    np.array
        Rotation (+scaling) matrix with shape 3x3
    np.array
        Translation vector with shape 3
    """

    npLandmarks = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    npTemplateLandmarks = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):

        npLandmarks[p,:3] = np.array(landmarks.GetPoint(p))
        npTemplateLandmarks[p,:] = np.array(templateLandmarks.GetPoint(p))

    
    R, t = RegisterPointClouds(npLandmarks, npTemplateLandmarks, scaling=scaling)

    center = np.mean(npLandmarks, axis=0).astype(np.float64)

    #transform = sitk.Similarity3DTransform()
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.ravel())
    transform.SetCenter(center)
    transform.SetTranslation(t)
    
    return transform

def RegisterSphericalMapImages(fixedImage, movingImage, mask, initial_transform=None, varianceForUpdateField=0.5, varianceForTotalField=10.0, fitSigmoid=False, verbose=False):
    """
    Calculates de transformation to align movingImage and fixedImage using dffeomorphic demons

    Parameters
    ----------
    fixedImage: sitkImage
        Fixed image
    movingImage: sitkImage
        Moving image
    initial_transform: sitkTransform
        Initial transform
    varianceForUpdateField: float
        Viscuosity
    varianceForTotalField: float
        Elasticity
    fitSigmoid:
        If True, image intensities are transformed to the range [0,1] using a sigmoid function
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    sitkTransform
        The transformation calculated
    """

    filter = sitk.CastImageFilter()
    filter.SetOutputPixelType(sitk.sitkFloat32)
    fixedImage = filter.Execute(fixedImage)

    filter = sitk.CastImageFilter()
    filter.SetOutputPixelType(sitk.sitkFloat32)
    movingImage = filter.Execute(movingImage)

    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    if initial_transform is None:
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixedImage)
        transform = sitk.Similarity2DTransform()
        displacementField = transform_to_displacment_field_filter.Execute(transform)
        initial_transform = sitk.DisplacementFieldTransform(displacementField)
    else:
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixedImage)
        displacementField = transform_to_displacment_field_filter.Execute(initial_transform)
        initial_transform = sitk.DisplacementFieldTransform(displacementField)
    
    # Transform (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=varianceForUpdateField, varianceForTotalField=varianceForTotalField)
    registration_method.SetInitialTransform(initial_transform)

    # Metric and mask
    registration_method.SetMetricAsDemons(0.5) #intensities are equal if the difference is less than 0.5

    ## Mask
    registration_method.SetMetricFixedMask(mask)
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2*fixedImage.GetSpacing()[0],fixedImage.GetSpacing()[0],0])

    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Fitting a sigmoid to the images to give more weight to the sutures during registration 
    if fitSigmoid:

        tmpArray = sitk.GetArrayFromImage(movingImage)
        minValue = np.min(tmpArray[sitk.GetArrayViewFromImage(mask) > 0])
        maxValue = np.max(tmpArray[sitk.GetArrayViewFromImage(mask) > 0])
        tmpArray = 250 / (1 + np.exp( (tmpArray - (minValue + (maxValue-minValue)/4.0) ) /  ((maxValue-minValue)/6.0)))
        tmpImage = sitk.GetImageFromArray(tmpArray)
        tmpImage.CopyInformation(movingImage)
        movingImage = tmpImage


        tmpArray = sitk.GetArrayFromImage(fixedImage)
        minValue = np.min(tmpArray[sitk.GetArrayViewFromImage(mask) > 0])
        maxValue = np.max(tmpArray[sitk.GetArrayViewFromImage(mask) > 0])
        #tmpArray = 1 / (1 + np.exp( 0.25 * (tmpArray - (maxValue-minValue)/2) / (maxValue-minValue) ))
        tmpArray = 250 / (1 + np.exp( (tmpArray - (minValue + (maxValue-minValue)/4.0) ) /  ((maxValue-minValue)/6.0)))
        tmpImage = sitk.GetImageFromArray(tmpArray)
        tmpImage.CopyInformation(fixedImage)
        fixedImage = tmpImage
        
    transform = registration_method.Execute(fixedImage, movingImage)

    return transform
