import os
from saveDatas import saveSqCo
squeeze_num = 4
data_num = len(os.listdir("./ERA5df"))
saveSqCo(squeeze_num, data_num)