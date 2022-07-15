# Normative Cranial Growth
This is a repository for the [Data-driven Normative Reference of Pediatric Cranial Bone Development](https://github.com/cuMIP/normativeCranialGrowth).
This repository contains the normative intensity (``IntensityModel``), thickness (``ThicknessModel``), and shape (``ShapeModel``) models described in the manuscript. This repository also provides the example scripts to generate cranial bone surface meshes based on age, sex, and standard deviation to the average principal component coefficients. The Excels files  average


![Network diagram as found in published manuscript](/ModelArchitecture.jpg)

## Dependencies:
- [Python](python.org)
- [Pytorch](https://pytorch.org/get-started/locally)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [VTK](https://pypi.org/project/vtk/)
- [scipy](https://scipy.org/)
- [skimage](https://scikit-image.org/)
- [TorchIO](https://torchio.readthedocs.io/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code
Due to data privacy agreements, we are not able to share any example CT Image. Our example inference codes are based on the data processing script to generate masked normalzied CT images based on MHA graphic data files (.mha). If you are not using the same data format for the CT images, you could generate input array based on your own choice. Our model requires input image arrays with size of 96x96x96 and intensity normalized to the range of 0-1.

### Quick summary
**Input**: MHA graphic files.

**Output**: MHA graphic files labeling 5 cranial bone, and VTK PolyData containing 4 landmarks at the cranial base.

### Code example
An example for automatic concurrent cranial bone labeling and landmark localization is:
```python
import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

### process example CT image

ctImage = sitk.ReadImage('./ExampleCTImage.mha')
binaryImage = DataProcessing.CreateBoneMask(ctImage)
ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)

### model
modelPath = './MiccaiFinalModel.dat'
device = ModelConfiguration.getDevice()
model = ModelConfiguration.adaptModel(modelPath, device)
imageData = ModelConfiguration.adaptData(ctImage, device)

landmarks, boneLabels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

```
*When using this code, be sure to assign corret path containing a valid MHA graphic file of a CT image to the ```ctImage```.*

### The workflow

- The **CreateBoneMask** function creates a binary mask for the CT image.
- The **ResampleAndMaskImage** function resamples and masks the CT image to the correct size and normalizes the intensity.
- The **getDevice** function specifies the device for the torch model.
- The **getDevice** function specifies the device for the torch model.
- The **adaptModel** and **adaptData** functions prepares the model and data for inference.
- The **runModel** function generates landmark and bone labeling predictions and resample them to the original CT image space.

If you have any questions, please email Jiawei Liu at jiawei.liu@cuanschutz.edu
