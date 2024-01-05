# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:45:13 2021

@author: lxw
"""

import os

# result = []
def get_all(cwd):
    get_dir = os.listdir(cwd)
    result = []
    for i in get_dir:
        sub_dir = os.path.join(cwd, i)
        result.append(sub_dir)
    return result


            
if __name__ == "__main__":
    root = r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData"
    postive_data = root + "\\PositiveData"
    positive_result = get_all(postive_data)
    positive_count = 0
    for i in positive_result:
        name = i.split('\\')[-1]
        name_part = name.split('.')
        img_name = "{0}.tif".format(name_part[0])
        shape_file_name = "{0}.shp".format(name_part[0])
        # print(name)
        # print(img_name,shape_file_name)
        with open(i,"r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(',')
                positive_left = line[1]
                positive_right = line[2]
                print(line[1],line[2])
                positive_count += 1
                break
            break
        
    print("positive data number: ", positive_count)
            
    # negative_result = get_all(r"F:\03Data\MyData\PhoenixCityGroup\PhoenixCityGroup\PhoenixCityGroup_BigImages\TrainingData\NegativeData")
    # negative_count = 0
    # for i in negative_result:
    #     name = i.split('\\')[-1]
        
    #     print(name) 
    #     with open(i,"r") as f:
    #         for line in f.readlines():
    #             line = line.strip('\n')
    #             # print(line)
    #             negative_count += 1
    # print("negative data number: ", negative_count)