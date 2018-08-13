import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# 한글 폰트 안 깨지게하기위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location='E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정 
font_name=fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

kovo_result_table = pd.read_pickle('Kovo_result_table')
kovo_part_table = pd.read_pickle('Kovo_part_table')

print(kovo_result_table)
print(kovo_part_table)

