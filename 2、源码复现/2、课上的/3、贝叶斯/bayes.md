### bayes1

【时间】11/12

* 别想着把理论学明白了再看代码，而是边看代码边学习需要的理论；
  总之，会用就行；

##### part1

``` python
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]      #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec
if __name__ == '__main__':
    postingLIst, classVec = loadDataSet()    
    for each in postingLIst:
        print(each)
    print(classVec)
```

【理解】

* postingList：对应的句子，每一个句子由词语组成；
* classVec：表示句子是否具有侮辱性；0表示无，1表示有；

【思考】

* 这样编码的好处是什么？😋

##### part2

``` python
# -*- coding: UTF-8 -*-

def setOfWords2Vec(vocabList, inputSet):
    # 创建返回向量数组长度个0
    # 集合单词
    returnVec = [0] * len(vocabList) 
    # 遍历输入的每一个句子的单词
    for word in inputSet:                                                
        if word in vocabList:      
            # 获取每一个单词所对应的索引，并在对应的位置压入1
            ## 比如，找到dog的索引为5，那么在5的位置就压入1；
            returnVec[vocabList.index(word)] = 1
        else: 
            # 否则就打印这里没有单词，同时这里的值设为1
            print("the word: %s is not in my Vocabulary!" % word)    
    return returnVec #返回文档向量

def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:               
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    print('postingList:\n',postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat = []    
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)
```

【步骤】

* 先加载数据集 -> 将所有数据转换为一个集合（集合特性：无序、无重复）-> 对数据进行编码

【代码解释】

``` python
# 1、对集合数组取交集和并集
a = set([1]) | set([2,3])
print(a)    # [1,2,3]
a = set([1,2]) & set([2,3])
print(a)    # [2]

# 2、数组乘法
a = [0] * 5  # [0,0,0,0,0]
```

【理解】

* 想办法找到测试的办法；
* 通过看main函数看测试顺序；
* 得到对应的词袋库；
* 对于向量最好拼成矩阵进行训练，速度较快；

##### part3

```  python
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(numWords);p1Num=np.zeros(numWords)
    p0Denom = 0.0;p1Denom=0.0    # 定义分母
    # 对于每一个训练的文档
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 这里加的是一个数组啊
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbusive
    
    
if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    # 这里打印了一下
    print('myVocabList:\n', myVocabList)
    trainMat = []    
    for postinDoc in postingList:
        # 得到对应的训练矩阵
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(len(trainMat))    # trainMat矩阵的长度
    print(trainMat[1])
    pOV = trainNB0(trainMat, classVec)
    # 将trainMat和类别传入trainNBO当中
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
```

【小白QA】

* Q1：TrainMatrix不是一个数组吗，为什么P1Num可以直接相加，然后做除法；
  A1：这就是模拟了贝叶斯公式，p1Num
* Q1：[1,0] + [0,1] 不是等于 [1,0,0,1]
  A1：确实是这样，但是这里是np.array；np.array：[1,0] + [0,1] = [1,1]
