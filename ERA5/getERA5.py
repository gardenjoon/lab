import cdsapi
import zipfile
from saveDatas import deleteFiles
import os

def callCdsApi(y, m, d):
    c = cdsapi.Client()

    c.retrieve(
        'satellite-sea-surface-temperature-ensemble-product',
        {
            'variable': 'all',
            'format': 'zip',
            'year': [y],
            'month': [m],
            'day': d
        },
        './ERA5zip/ERA5'+y+m+d[0]+d[-1]+'.zip'
    )

def unzip(path):
    file_list = os.listdir(path)
    file_name = file_list[0]
    zipfile.ZipFile(path+file_name).extractall(path=path+'../ERA5nc')
    deleteFiles('./ERA5zip/')
