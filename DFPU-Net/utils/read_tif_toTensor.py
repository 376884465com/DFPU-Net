from osgeo import gdal
from torchvision.transforms import functional as F


class ReadTiff(object):
    def readTif_to_Ndarray(self, img_path):
        dataset = gdal.Open(img_path)
       
        self.width = dataset.RasterXSize
        
        self.height = dataset.RasterYSize
        
        self.data = dataset.ReadAsArray(0, 0, self.width, self.height)
        return self.data

    def Ndarray_to_Tensor(self, ndarray):
        
        F.to_tensor(ndarray).permute(1, 0, 2)
        return Tensor

