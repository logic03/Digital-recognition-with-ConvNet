import os
b = 0
dir = 'F:/ele/data/'
#os.listdir的结果就是一个list集合
#可以使用一个list的sort方法进行排序，有数字就用数字排序
files = os.listdir(dir)
files.sort()
#print("files:", files)
train = open('F:/ele/data/train.txt', 'a')
test = open('F:/ele/data/test.txt', 'a')
a = 0
a1 = 0
while(b < 20):#20是因为10个train文件夹+10个valid的文件夹
    #这里采用的是判断文件名的方式进行处理
    if 'train' in files[b]:#如果文件名有train
        label = a #设置要标记的标签，比如sample001_train里面都是0的图片，标签就是0
        ss = 'F:/ele/data/' + str(files[b]) + '/' #700张训练图片
        pics = os.listdir(ss) #得到sample001_train文件夹下的700个文件名
        i = 1
        while i < 701:#一共有700张
            name = str(dir) + str(files[b]) + '/' + pics[i-1] + ' ' + str(int(label)) + '\n'
            train.write(name)
            i = i + 1
        a = a + 1
    if 'valid' in files[b]:
        label = a1
        ss = 'F:/ele/data/' + str(files[b]) + '/' #200张验证图片
        pics = os.listdir(ss)
        j = 1
        while j < 201:
            name = str(dir) + str(files[b]) + '/' + pics[j-1] + ' ' + str(int(label)) + '\n'
            test.write(name)
            j = j + 1
        a1 = a1 + 1
    b = b + 1

    '''
            if i < 701:
                fileType = os.path.split(file)
                if fileType[1] == '.txt':
                    continue
                name = str(dir) + file + ' ' + str(int(label)) + '\n'
                train.write(name)
                i = i + 1
    '''
