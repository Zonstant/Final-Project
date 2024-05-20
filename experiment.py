import os
import numpy as np
import random
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,confusion_matrix,classification_report
import pickle
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import multiprocessing
from flpSrc import *


def getFeature1(f,c):
    c+=1
    matrix=len(c)*f*c/np.sum(c.T,axis=1)+c
    matrix=1/matrix.T
    weight=np.sum(c,axis=1)/np.sum(c)
    matrix=np.array(matrix)*np.array(weight)
    feature1=np.sum(matrix,axis=1)
    feature1=(feature1-np.mean(feature1))/np.std(feature1)
    return feature1

def getFeature2(f,c):
    m=len(f)
    n=len(c)
    y=np.zeros((m,n))
    cSorted=np.argsort(c,axis=1)
    for i in range(n):
        for j in range(m):
            y[cSorted[i][j]][i]=1
            if j>np.sqrt(n):
                break
    weight=np.sum(c,axis=1)/np.sum(c)
    newlac=np.zeros(m)
    for j in range(m):
        if np.sum(y[j])!=0:
            cost=f[j]+np.sum(c.T[j]*y[j])
            newlac[j]=cost/np.sum(weight*y[j])
    for j in range(m):
        if np.sum(y[j])==0:
            newlac[j]=np.max(newlac)
    feature2=np.array(newlac)
    feature2=(feature2-np.mean(feature2))/np.std(feature2)
    return feature2

def getFeature3(f,c):
    feature1=getFeature1(f,c)
    weight=(feature1/c).T
    for i,j in np.argwhere(weight<0):
        weight[i][j]=0
    weight=1/np.sum(weight.T,axis=1)
    c+=1
    matrix=len(c)*f*c/np.sum(c.T,axis=1)+c
    matrix=1/matrix.T
    matrix=np.array(matrix)*np.array(weight)
    feature3=np.sum(matrix,axis=1)
    feature3=(feature3-np.mean(feature3))/np.std(feature3)
    return feature3

def getFeatures(f,c):
    m=len(f)
    n=len(c)

    #Feature1
    c+=1
    matrix=len(c)*f*c/np.sum(c.T,axis=1)+c
    matrix=1/matrix.T
    weight=np.sum(c,axis=1)/np.sum(c)
    matrix=np.array(matrix)*np.array(weight)
    feature1=np.sum(matrix,axis=1)
    feature1=(feature1-np.mean(feature1))/np.std(feature1)

    #Feature2
    y=np.zeros((m,n))
    cSorted=np.argsort(c,axis=1)
    for i in range(n):
        for j in range(m):
            y[cSorted[i][j]][i]=1
            if j>np.sqrt(n):
                break
    newlac=np.zeros(m)
    for j in range(m):
        if np.sum(y[j])!=0:
            cost=f[j]+np.sum(c.T[j]*y[j])
            newlac[j]=cost/np.sum(weight*y[j])
    for j in range(m):
        if np.sum(y[j])==0:
            newlac[j]=np.max(newlac)
    feature2=np.array(newlac)
    feature2=(feature2-np.mean(feature2))/np.std(feature2)

    #Feature3
    weight=(feature1/c).T
    for i,j in np.argwhere(weight<0):
        weight[i][j]=0
    weight=1/np.sum(weight.T,axis=1)
    c+=1
    matrix=len(c)*f*c/np.sum(c.T,axis=1)+c
    matrix=1/matrix.T
    matrix=np.array(matrix)*np.array(weight)
    feature3=np.sum(matrix,axis=1)
    feature3=(feature3-np.mean(feature3))/np.std(feature3)

    features=[[] for i in range(len(f))]
    for j in range(len(f)):
        features[j].append(feature1[j])
        features[j].append(feature2[j])
        features[j].append(feature3[j])
    return features

def train(feature,label,check):
    #划分训练集与测试集9：1
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, random_state=42)

    # # 将连续值标签转换为离散值标签
    # y_train= np.where( y_train > 0.5, 1, 0)
    # y_test= np.where(y_test > 0.5, 1, 0)

    # 使用RandomOverSampler处理不平衡数据
    #过采样使得少数类与多数类比例提升到1:2

    oversample = RandomOverSampler(sampling_strategy=0.5)
    if check:
        X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
  
    # 建立XGBoost分类模型
    # max_depth: 树的深度，默认值是6，值过大容易过拟合，值过小容易欠拟合。
    xgb=XGBClassifier(max_depth=24, random_state=42)
    scores = cross_val_score(xgb, X_train_resampled, y_train_resampled, cv=5)

    # 输出交叉验证结果
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(np.mean(scores)))

    # 使用测试数据集评估模型性能
    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred = xgb.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #输出分类的预测概率
    y_proba = xgb.predict_proba(X_test)[:, 1]

    #检验是否过拟合
    #预测训练集数据
    # y_check = xgb.predict(X_train)
    # print(confusion_matrix(y_train, y_check))
    # print(classification_report(y_train, y_check))

    #保存训练的模型
    pickle.dump(xgb,open('./modelLACfromXGBClassifier.sav','wb'))
    return xgb,confusion_matrix(y_test, y_pred),classification_report(y_test, y_pred)


def trainRandom(dataset,featureMethod,tt,name):
    if dataset is None:
        #generate instances
        dirpath='./experiments/train/Random/'+str(tt)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for t in range(3000):
            f,c=generateRandom([5,30],True)
            label,value=getLabel(f,c)
            writeRandom(f,c,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
    else:
        dirpath=dataset+str(tt)
    #read instances
    feature=[]
    label=[]
    for t in range(3000):
        f,c,x=readRandom(dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
        feature=feature+featureMethod(np.array(f),np.array(c))
        label.extend(x)
    #train
    #有imbalance
    feature=np.array(feature)
    label=np.array(label)
    print(feature.shape)
    print(label.shape)
    model,confusion,report=train(feature,label,False)
    file=open(dirpath+'/'+name+'/confusion.txt','w')
    for i in confusion:
        for j in i:
            file.write(str(j)+' ')
        file.write('\n')
    file.close()
    file=open(dirpath+'/'+name+'/report.txt','w')
    file.write(report)
    file.close()
    pickle.dump(model,open(dirpath+'/'+name+'/model.sav','wb'))

def trainOR(dataset,featureMethod,tt,name):
    if dataset is None:
        #generate instances
        dirpath='./experiments/train/OR/'+str(tt)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for t in range(3000):
            m=random.randint(20,30)
            n=m*random.randint(1,10)
            f,c,d=generateOR(m,n,np.random.randint(4)+3)
            label,value=getLabel(f,c)
            writeOR(f,c,d,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
    else:
        dirpath=dataset+str(tt)
    #read instances
    feature=[]
    label=[]
    for t in range(3000):
        f,c,x=readOR(dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
        feature=feature+featureMethod(np.array(f),np.array(c))
        label.extend(x)
    #train
    #有imbalance
    feature=np.array(feature)
    label=np.array(label)
    print(feature.shape)
    print(label.shape)
    model,confusion,report=train(feature,label,False)
    file=open(dirpath+'/'+name+'/confusion.txt','w')
    for i in confusion:
        for j in i:
            file.write(str(j)+' ')
        file.write('\n')
    file.close()
    file=open(dirpath+'/'+name+'/report.txt','w')
    file.write(report)
    file.close()
    pickle.dump(model,open(dirpath+'/'+name+'/model.sav','wb'))

def trainM(dataset,featureMethod,tt,name):
    if dataset is None:
        #generate instances
        dirpath='./experiments/train/M/'+str(tt)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for t in range(3000):
            print(tt,t)
            m=random.randint(5,30)
            n=m*random.randint(1,10)
            f,c,b=generateM(m,n,m/2,m*3,0.2,1,1,5)
            label,value=getLabel(f,c)
            writeM(f,c,b,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
    else:
        dirpath=dataset+str(tt)
    #read instances
    feature=[]
    label=[]
    for t in range(3000):
        f,c,x=readM(dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
        feature=feature+featureMethod(np.array(f),np.array(c))
        label.extend(x)
    #train
    #有imbalance
    model,confusion,report=train(feature,label,False)
    file=open(dirpath+'/'+name+'/confusion.txt','w')
    for i in confusion:
        for j in i:
            file.write(str(j)+' ')
        file.write('\n')
    file.close()
    file=open(dirpath+'/'+name+'/report.txt','w')
    file.write(report)
    file.close()
    pickle.dump(model,open(dirpath+'/'+name+'/model.sav','wb'))

def trainGap(dataset,featureMethod,tt,name):
    if dataset is None:
        #generate instances
        dirpath='./experiments/train/Gap/'+str(tt)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for t in range(3000):
            print(tt,t)
            m=random.randint(20,30)
            l=np.random.randint(4)+5
            if t<1000:
                f,c=generateGap(m,3000,30034,l,'A')
            elif t<2000:
                f,c=generateGap(m,3000,30034,l,'B')
            else:
                f,c=generateGap(m,3000,30034,l,'C')
            label,value=getLabel(f,c)
            writeGap(f,c,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')
    else:
        dirpath=dataset+str(tt)
    #read instances
    feature=[]
    label=[]
    for t in range(3000):
        f,c,x=readGap(dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt',False)
        feature=feature+featureMethod(np.array(f),np.array(c))
        label.extend(x)
    #train
    #有imbalance
    model,confusion,report=train(feature,label,False)
    file=open(dirpath+'/'+name+'/confusion.txt','w')
    for i in confusion:
        for j in i:
            file.write(str(j)+' ')
        file.write('\n')
    file.close()
    file=open(dirpath+'/'+name+'/report.txt','w')
    file.write(report)
    file.close()
    pickle.dump(model,open(dirpath+'/'+name+'/model.sav','wb'))

def experiment_large(f,c,name,feCheck,model,t,seed,kcheck,dynamic):
    #large M instances
    print(name,len(f),t,seed)
    random.seed(seed)
    np.random.seed(seed)
    dirpath='./experiments/results/'+str(t)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)  
    if len(f)==1000:
        size=30
        fe_count=7000
        time_count=4
    if  len(f)==2000:
        size=50
        fe_count=15000
        time_count=40
    if  len(f)==3000:
        size=70
        fe_count=22000
        time_count=180
    if  len(f)==5000:
        size=90
        fe_count=26000
        time_count=800
    if  len(f)==10000:
        size=100
        fe_count=51000
        time_count=1200
    if feCheck:
        a='fe'
        time_count=-1
    else:
        a='time'
        fe_count=-1
    if model is None:
        b='original'
    else:
        model=pickle.load(open(model,'rb'))
        b='model'
    if dynamic:
        d='dynamic'
    elif kcheck:
        d='directwithK'
    else:
        d='direct'

    # feature=getFeatures(f,c)
    # predict=model.predict(feature)
    # fs=[]
    # for j in range(len(f)):
    #     if predict[j]==1:
    #         fs.append(j)
    # print(str(len(f))+' model reduce:'+str(len(fs)))
    from DPSO import Main

    file=open(dirpath+'/'+a+'_'+b+'_'+d+'_'+name+'.txt','w')
    if model is None:
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed}\n')
        file.flush()
        main_instance = Main(f,c,size,fe_count,time_count)
        solution,best = main_instance.DPSO(file)
    elif dynamic:
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed}\n')
        main_instance = Main(f,c,size,fe_count,time_count)
        solution,best = main_instance.DPSO_dynamic(file,model,5,1)
    elif kcheck:
        feature=getFeatures(f,c)
        predict=model.predict(feature)
        indexes=np.argsort(-1*predict)
        main_instance = Main(f,c,size,fe_count,time_count)
        k=main_instance.findK(indexes)
        main_instance.f=main_instance.f[indexes[:k]]
        main_instance.c=main_instance.c[:,indexes[:k]]
        main_instance.m=len(main_instance.f)
        main_instance.n=len(main_instance.c)
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed} {k}\n')
        file.flush()
        solution,best = main_instance.DPSO(file)

    else:
        feature=getFeatures(f,c)
        predict=model.predict(feature)
        fs=[]
        for j in range(len(f)):
            if predict[j]==1:
                fs.append(j)
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed} {len(fs)}\n')
        file.flush()
        main_instance = Main(f[fs],c[:,fs],size,fe_count,time_count)
        solution,best = main_instance.DPSO(file)
    for xx in solution:
        file.write(str(xx)+' ')
    file.close()

def experiment_large_name(g,filename,name,feCheck,model,t,seed,kcheck,dynamic):
    #large M instances
    if g =='Random':
        f,c,x=readRandom(filename,None)
    if g=='M':
        f,c,x=readM(filename,None)
    if g=='OR':
        f,c,x=readOR(filename,None)
    if g=='Gap':
        f,c,x=readGap(filename,None,False)
    f=np.array(f)
    c=np.array(c)
    print(name,len(f),t,seed)
    random.seed(seed)
    np.random.seed(seed)
    dirpath='./experiments/results/'+str(t)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)  
    if len(f)==1000:
        size=30
        fe_count=7000
        time_count=4
    if  len(f)==2000:
        size=50
        fe_count=15000
        time_count=40
    if  len(f)==3000:
        size=70
        fe_count=22000
        time_count=180
    if  len(f)==5000:
        size=90
        fe_count=26000
        time_count=800
    if  len(f)==10000:
        size=100
        fe_count=51000
        time_count=1200
    else:
        print(len(f),'wrong')
    if feCheck:
        a='fe'
        time_count=-1
    else:
        a='time'
        fe_count=-1
    if model is None:
        b='original'
    else:
        model=pickle.load(open(model,'rb'))
        b='model'
    if dynamic:
        d='dynamic'
    elif kcheck:
        d='directwithK'
    else:
        d='direct'

    # feature=getFeatures(f,c)
    # predict=model.predict(feature)
    # fs=[]
    # for j in range(len(f)):
    #     if predict[j]==1:
    #         fs.append(j)
    # print(str(len(f))+' model reduce:'+str(len(fs)))
    from DPSO import Main

    file=open(dirpath+'/'+a+'_'+b+'_'+d+'_'+name+'.txt','w')
    if model is None:
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed}\n')
        file.flush()
        main_instance = Main(f,c,size,fe_count,time_count)
        solution,best = main_instance.DPSO(file)
    elif dynamic:
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed}\n')
        main_instance = Main(f,c,size,fe_count,time_count)
        solution,best = main_instance.DPSO_dynamic(file,model,5,1)
    elif kcheck:
        feature=getFeatures(f,c)
        predict=model.predict(feature)
        indexes=np.argsort(-1*predict)
        main_instance = Main(f,c,size,fe_count,time_count)
        k=main_instance.findK(indexes)
        main_instance.f=main_instance.f[indexes[:k]]
        main_instance.c=main_instance.c[:,indexes[:k]]
        main_instance.m=len(main_instance.f)
        main_instance.n=len(main_instance.c)
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed} {k}\n')
        file.flush()
        solution,best = main_instance.DPSO(file)

    else:
        feature=getFeatures(f,c)
        predict=model.predict(feature)
        fs=[]
        for j in range(len(f)):
            if predict[j]==1:
                fs.append(j)
        file.write(f'{name} {len(f)} {fe_count} {time_count} {size} {seed} {len(fs)}\n')
        file.flush()
        main_instance = Main(f[fs],c[:,fs],size,fe_count,time_count)
        solution,best = main_instance.DPSO(file)
    for xx in solution:
        file.write(str(xx)+' ')
    file.close()

def readresults(filename,model,dynamic):
    file=open(filename)
    s=file.readline()
    ss=s.split(' ')
    name=ss[0]
    size=eval(ss[1])
    rsize=None
    if model:
        rsize=eval(ss[6])
    s=file.readline()
    fitness=[]
    fe_count=[]
    time_count=[]
    sizes=[]
    while s!='':
        ss=s.split(' ')
        if len(ss)>10:
            x=[]
            for xx in ss:
                if xx!='':
                    x.append(eval(xx))
            #print(name,round(np.sum(x)/len(x),3))
        else:
            if dynamic:
                if len(ss)==7:
                    sizes.append(eval(ss[4]))
            fitness.append(eval(ss[1]))
            fe_count.append(eval(ss[2]))
            time_count.append(eval(ss[3]))
        s=file.readline()
    return name,size,rsize,fitness,fe_count,time_count,sizes

if __name__=='__main__':

    #因为是UFLP,再将未完之后做搜索得到的解，作比较时不用放回原问题再算fitness_value，如果删掉的是customer，则要放回计算
    #collect data
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists('./experiments/train'):
        os.mkdir('./experiments/train')
    if not os.path.exists('./experiments/test'):
        os.mkdir('./experiments/test')

    dirpath='./experiments/train/Random'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    processes=[]
    for tt in range(12):
        process=multiprocessing.Process(target=trainRandom,args=(None,getFeatures,tt,))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    processes=[]
    for tt in range(12,25):
        process=multiprocessing.Process(target=trainRandom,args=(None,getFeatures,tt,))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()

    dirpath='./experiments/train/OR'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    processes=[]
    for tt in range(12):
        process=multiprocessing.Process(target=trainOR,args=(None,getFeatures,tt,))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    processes=[]
    for tt in range(12,25):
        process=multiprocessing.Process(target=trainOR,args=(None,getFeatures,tt,))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    
    dirpath='./experiments/train/M'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(25):
        trainM(None,getFeatures,t)
    
    dirpath='./experiments/train/Gap'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(25):
        trainGap(None,getFeatures,t)

    dirpath='./experiments/train/Random/'
    for t in range(25):
        if not os.path.exists(dirpath+str(t)+'/features'):
            os.mkdir(dirpath+str(t)+'/features')
        trainRandom(dirpath,getFeatures,t,'features')
    
    dirpath='./experiments/train/OR/'
    for t in range(25):
        if not os.path.exists(dirpath+str(t)+'/features'):
            os.mkdir(dirpath+str(t)+'/features')
        trainOR(dirpath,getFeatures,t,'features')
    
    dirpath='./experiments/train/M/'
    for t in range(25):
        if not os.path.exists(dirpath+str(t)+'/features'):
            os.mkdir(dirpath+str(t)+'/features')
        trainM(dirpath,getFeatures,t,'features')
    
    dirpath='./experiments/train/Gap/'
    for t in range(25):
        if not os.path.exists(dirpath+str(t)+'/features'):
            os.mkdir(dirpath+str(t)+'/features')
        trainGap(dirpath,getFeatures,t,'features')

    
    #generate instances
    dirpath='./experiments/test/Random'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(10):
        f,c=generateRandomTest(100,100,True)
        label,value=getLabel(f,c)
        writeRandom(f,c,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')

    #generate instances
    dirpath='./experiments/test/OR'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(10):
        f,c,d=generateOR(100,100,6)
        label,value=getLabel(f,c)
        writeOR(f,c,d,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')

    #generate instances
    dirpath='./experiments/test/M'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(3000):
        m=random.randint(5,30)
        n=m*random.randint(1,10)
        f,c,b=generateM(100,100,50,300,4,20,1,5)
        label,value=getLabel(f,c)
        writeM(f,c,b,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')

    #generate instances
    dirpath='./experiments/test/Gap'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for t in range(10):
        m=100
        l=30
        if t<3:
            f,c=generateGap(m,3000,30034,l,'A')
        elif t<6:
            f,c=generateGap(m,3000,30034,l,'B')
        else:
            f,c=generateGap(m,3000,30034,l,'C')
        label,value=getLabel(f,c)
        writeGap(f,c,label,value,dirpath+'/instances'+str(t)+'.txt',dirpath+'/instances_sol'+str(t)+'.txt')

    #Experiment1:train
    precision0=[]
    recall0=[]
    f1score0=[]
    precision1=[]
    recall1=[]
    f1score1=[]
    for i in range(25):
        dirpath='./experiments/train/Random/'+str(i)+'/features/'
        file=open(dirpath+'report.txt')
        s=file.readline()
        s=file.readline()
        s=file.readline()
        if '0.0' in s:
            count=0
            for ss in s.replace('\n','').split(' '):
                if ss!='':
                    if count==1:
                        precision0.append(eval(ss))
                    elif count==2:
                        recall0.append(eval(ss))
                    elif count==3:
                        f1score0.append(eval(ss))
                    count+=1
            if count!=5:
                print('count wrong1')
        else:
            print('wrong!')
        s=file.readline()
        if '1.0' in s:
            count=0
            for ss in s.replace('\n','').split(' '):
                if ss!='':
                    if count==1:
                        precision1.append(eval(ss))
                    elif count==2:
                        recall1.append(eval(ss))
                    elif count==3:
                        f1score1.append(eval(ss))
                    count+=1
            if count!=5:
                print('count wrong1')
        else:
            print('wrong!')
    s=''
    s=s+str(np.round(np.average(precision0),2))+'\pm'+str(np.round(np.std(precision0),2))+' & '
    s=s+str(np.round(np.average(precision1),2))+'\pm'+str(np.round(np.std(precision1),2))+' & '
    s=s+str(np.round(np.average(recall0),2))+'\pm'+str(np.round(np.std(recall0),2))+' & '
    s=s+str(np.round(np.average(recall1),2))+'\pm'+str(np.round(np.std(recall1),2))+' & '
    s=s+str(np.round(np.average(f1score0),2))+'\pm'+str(np.round(np.std(f1score0),2))+' & '
    s=s+str(np.round(np.average(f1score1),2))+'\pm'+str(np.round(np.std(f1score1),2))+' \\\\'
    print(s)

    #Experiment1:test
    precision0=[]
    recall0=[]
    f1score0=[]
    precision1=[]
    recall1=[]
    f1score1=[]
    for i in range(25):
        model=pickle.load(open('./experiments/train/Gap/'+str(i)+'/features/model.sav','rb'))
        y_pred=[]
        y_test=[]
        for t in range(10):
            dirpath='./experiments/test/Gap'
            filname=dirpath+'/instances'+str(t)+'.txt'
            solname=dirpath+'/instances_sol'+str(t)+'.txt'
            f,c,x=readGap(filname,solname,False)
            f=np.array(f)
            c=np.array(c)
            y_pred+=model.predict(getFeatures(f,c)).tolist()
            y_test+=x
        file=open('./experiments/train/Gap/'+str(i)+'/features/report_test.txt','w')
        file.write(classification_report(y_test, y_pred))
        file.close()
        file=open('./experiments/train/Gap/'+str(i)+'/features/report_test.txt')
        s=file.readline()
        s=file.readline()
        s=file.readline()
        if '0.0' in s:
            count=0
            for ss in s.replace('\n','').split(' '):
                if ss!='':
                    if count==1:
                        precision0.append(eval(ss))
                    elif count==2:
                        recall0.append(eval(ss))
                    elif count==3:
                        f1score0.append(eval(ss))
                    count+=1
            if count!=5:
                print('count wrong1')
        else:
            print('wrong!')
        s=file.readline()
        if '1.0' in s:
            count=0
            for ss in s.replace('\n','').split(' '):
                if ss!='':
                    if count==1:
                        precision1.append(eval(ss))
                    elif count==2:
                        recall1.append(eval(ss))
                    elif count==3:
                        f1score1.append(eval(ss))
                    count+=1
            if count!=5:
                print('count wrong1')
        else:
            print('wrong!')
    s=''
    s=s+str(np.round(np.average(precision0),2))+'\pm'+str(np.round(np.std(precision0),2))+' & '
    s=s+str(np.round(np.average(precision1),2))+'\pm'+str(np.round(np.std(precision1),2))+' & '
    s=s+str(np.round(np.average(recall0),2))+'\pm'+str(np.round(np.std(recall0),2))+' & '
    s=s+str(np.round(np.average(recall1),2))+'\pm'+str(np.round(np.std(recall1),2))+' & '
    s=s+str(np.round(np.average(f1score0),2))+'\pm'+str(np.round(np.std(f1score0),2))+' & '
    s=s+str(np.round(np.average(f1score1),2))+'\pm'+str(np.round(np.std(f1score1),2))+' \\\\'
    print(s)

    # Experiment2
    rsize=[[] for i in range(10)]
    rgap=[[] for i in range(10)]
    index=[[] for i in range(10)]
    openp=[0 for i in range(10)]
    for i in range(25):
        model=pickle.load(open('./experiments/train/Random/'+str(i)+'/features/model.sav','rb'))
        for t in range(10):
            dirpath='./experiments/test/Random'
            filname=dirpath+'/instances'+str(t)+'.txt'
            solname=dirpath+'/instances_sol'+str(t)+'.txt'
            f,c,x=readRandom(filname,solname)
            openp[t]=np.sum(x)/len(f)
            f=np.array(f)
            c=np.array(c)
            best=functionValue(f,c,x)
            feature=getFeatures(f,c)
            predict=model.predict(feature)
            fs=[]
            for j in range(len(f)):
                if predict[j]==1:
                    fs.append(j)
            ff=f[fs]
            cc=c[:,fs]
            rsize[t].append(len(ff)/len(f))
            label,value=getLabel(ff,cc)
            rgap[t].append((value-best)/best*100)
            predict=model.predict_proba(feature)[:,1]
            indexes=np.argsort(-1*predict).tolist()
            for j in range(len(x)):
                if x[j]==1:
                    index[t].append((indexes.index(j)+1)/len(x))

    s='\\hline 最优解仓库比例&'
    for t in range(10):
        s=s+str(openp[t])+' &'
    s=s[:-1]
    s=s+'\\\\'
    print(s)

    s='\\hline 变量保留比例&'
    for t in range(10):
        s=s+str(np.round(np.average(rsize[t]),2))+'\pm'+str(np.round(np.std(rsize[t]),2))+' &'
    s=s[:-1]
    s=s+'\\\\'
    print(s)

    s='\\hline 最优解差异&'
    for t in range(10):
        s=s+str(np.round(np.average(rgap[t]),2))+'\pm'+str(np.round(np.std(rgap[t]),2))+' \\% &'
    s=s[:-1]
    s=s+'\\\\'
    print(s)

    s='\\hline 最优解仓库平均排名&'
    for t in range(10):
        s=s+str(np.round(np.average(index[t]),2))+'\pm'+str(np.round(np.std(index[t]),2))+' &'
    s=s[:-1]
    s=s+'\\\\'
    print(s)

    #Experiment 3,4

    for size in [2000,3000,5000,10000]:
        for p in [0.1,0.01,0.2]:
            for g in ['OR','M','Random']:
                for i in range(3):
                    if size==10000 and i==1:
                        break
                    print(p,g,size)
                    if g =='Random':
                        f,c,x=readRandom('experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',None)
                    if g=='M':
                        f,c,x=readM('experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',None)
                    if g=='OR':
                        f,c,x=readOR('experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',None)
                    if g=='Gap':
                        f,c,x=readGap('experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',None,False)
                    f=np.array(f)
                    c=np.array(c)

                    processes=[]
                    for t in range(15):
                        seed = random.randint(1, 1000000)
                        model='./experiments/train/'+g+'/'+str(t)+'/features/model.sav'
                        process=multiprocessing.Process(target=experiment_large,args=(f,c,g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,None,t,seed,False,False,))
                        process.start()
                        processes.append(process)
                    for process in processes:
                        process.join()

                    processes=[]
                    for t in range(15):
                        seed = random.randint(1, 1000000)
                        model='./experiments/train/'+g+'/'+str(t)+'/features/model.sav'
                        process=multiprocessing.Process(target=experiment_large,args=(f,c,g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,False,False,))
                        process.start()
                        processes.append(process)
                    for process in processes:
                        process.join()

                    processes=[]
                    for t in range(15):
                        seed = random.randint(1, 1000000)
                        model='./experiments/train/'+g+'/'+str(t)+'/features/model.sav'
                        process=multiprocessing.Process(target=experiment_large,args=(f,c,g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,True,False,))
                        process.start()
                        processes.append(process)
                    for process in processes:
                        process.join()

                    processes=[]
                    for t in range(15):
                        seed = random.randint(1, 1000000)
                        model='./experiments/train/'+g+'/'+str(t)+'/features/model.sav'
                        process=multiprocessing.Process(target=experiment_large,args=(f,c,g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,False,True,))
                        process.start()
                        processes.append(process)
                    for process in processes:
                        process.join()
    
    # from multiprocessing import Pool
    # pool = Pool(15)
    # for p,g,size,t,runkind in instances:
    #     i=0
    #     seed = random.randint(1, 1000000)
    #     model='./experiments/train/'+g+'/'+str(t)+'/features/model.sav'
    #     if runkind=='direct':
    #         pool.apply_async(func=experiment_large_name,args=(g,'experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,False,False,))
    #     elif runkind=='directwithK':
    #         pool.apply_async(func=experiment_large_name,args=(g,'experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,True,False,))
    #     elif runkind=='dynamic':
    #         pool.apply_async(func=experiment_large_name,args=(g,'experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,model,t,seed,False,True,))
    #     elif runkind=='original':
    #         pool.apply_async(func=experiment_large_name,args=(g,'experiments/test/p='+str(p)+'/'+g+'_'+str(size)+'_'+str(i)+'.txt',g+'_'+str(size)+'_'+str(i)+'_'+str(p),True,None,t,seed,False,False,))
    #     else:
    #         print(p,g,size,t,runkind)
    # pool.close()
    # pool.join()
    

    for p in [0.01,0.1,0.2]:
        for g in ['Random','M','Gap']:
            print(p,g)
            count=0
            s=''
            for ssize in [2000]:
                for i in range(3):
                    # if ssize in [5000,10000] and i==1:
                    #     break
                    count+=1
                    finalp=[]
                    obest=[]
                    osizes=[]
                    otime=[]
                    rbest=[]
                    rsizes=[]
                    rtime=[]
                    kbest=[]
                    ksizes=[]
                    ktime=[]
                    dbest=[]
                    dsizes=[]
                    dsizee=[]
                    dtime=[]
                    for t in range(15):
                        name,size,rsize,fitness,fe_count,time_count,sizes=readresults('./experiments/results/'+str(t)+'/fe_original_direct_'+g+'_'+str(ssize)+'_'+str(i)+'_'+str(p)+'.txt',False,False)
                        obest.append(fitness[-1:][0])
                        osizes.append(size)
                        otime.append(time_count[-1:][0])
                        name,size,rsize,fitness,fe_count,time_count,sizes=readresults('./experiments/results/'+str(t)+'/fe_model_direct_'+g+'_'+str(ssize)+'_'+str(i)+'_'+str(p)+'.txt',True,False)
                        rbest.append(fitness[-1:][0])
                        rsizes.append(rsize)
                        rtime.append(time_count[-1:][0])
                        name,size,rsize,fitness,fe_count,time_count,sizes=readresults('./experiments/results/'+str(t)+'/fe_model_directwithK_'+g+'_'+str(ssize)+'_'+str(i)+'_'+str(p)+'.txt',True,False)
                        kbest.append(fitness[-1:][0])
                        ksizes.append(rsize)
                        ktime.append(time_count[-1:][0])
                        name,size,rsize,fitness,fe_count,time_count,sizes=readresults('./experiments/results/'+str(t)+'/fe_model_dynamic_'+g+'_'+str(ssize)+'_'+str(i)+'_'+str(p)+'.txt',False,True)
                        dbest.append(fitness[-1:][0])
                        dsizes.append(sizes[0])
                        dsizee.append(sizes[:-1][0])
                        dtime.append(time_count[-1:][0])
                    import scipy.stats as stats
                    s=s+'\\hline '+g+f'{count} & {round(np.average(osizes))} &'
                    s1,p1 = stats.wilcoxon(obest,rbest,alternative='less')
                    s2,p2 = stats.wilcoxon(obest,kbest,alternative='less')
                    s3,p3 = stats.wilcoxon(obest,dbest,alternative='less')
                    if p1<=0.05 and p2<=0.05 and p3<=0.05:
                        s=s+'\\textbf{'+str(np.round(np.average(obest),3))+'$\\pm$'+str(np.round(np.std(obest),3))+'}'
                    else:
                        s=s+str(np.round(np.average(obest),3))+'$\\pm$'+str(np.round(np.std(obest),3))
                    s=s+f'&{round(np.average(otime),1)}&{round(np.average(rsizes))}&'
                    s1,p1 = stats.wilcoxon(rbest,obest,alternative='less')
                    s2,p2 = stats.wilcoxon(rbest,kbest,alternative='less')
                    s3,p3 = stats.wilcoxon(rbest,dbest,alternative='less')
                    if p1<=0.05 and p2<=0.05 and p3<=0.05:
                        s=s+'\\textbf{'+str(np.round(np.average(rbest),3))+'$\\pm$'+str(np.round(np.std(rbest),3))+'}'
                    else:
                        s=s+str(np.round(np.average(rbest),3))+'$\\pm$'+str(np.round(np.std(rbest),3))
                    s=s+f'&{round(np.average(rtime),1)}&{round(np.average(ksizes))}&'
                    s1,p1 = stats.wilcoxon(kbest,obest,alternative='less')
                    s2,p2 = stats.wilcoxon(kbest,rbest,alternative='less')
                    s3,p3 = stats.wilcoxon(kbest,dbest,alternative='less')
                    if p1<=0.05 and p2<=0.05 and p3<=0.05:
                        s=s+'\\textbf{'+str(np.round(np.average(kbest),3))+'$\\pm$'+str(np.round(np.std(kbest),3))+'}'
                    else:
                        s=s+str(np.round(np.average(kbest),3))+'$\\pm$'+str(np.round(np.std(kbest),3))
                    s=s+f'&{round(np.average(ktime),1)}&{round(np.average(dsizes))}-{round(np.average(dsizee))}&'
                    s1,p1 = stats.wilcoxon(dbest,obest,alternative='less')
                    s2,p2 = stats.wilcoxon(dbest,rbest,alternative='less')
                    s3,p3 = stats.wilcoxon(dbest,kbest,alternative='less')
                    if p1<=0.05 and p2<=0.05 and p3<=0.05:
                        s=s+'\\textbf{'+str(np.round(np.average(dbest),3))+'$\\pm$'+str(np.round(np.std(dbest),3))+'}'
                    else:
                        s=s+str(np.round(np.average(dbest),3))+'$\\pm$'+str(np.round(np.std(dbest),3))
                    s=s+f'&{round(np.average(dtime),1)}\\\\ \n'
            print(s)

    import matplotlib.pyplot as plt
    for p in [0.01,0.1,0.2]:
        for g in ['Random','M','Gap']:
            for ssize in [2000]:
                name,size,rsize,fitness0,fe_count0,time_count,sizes=readresults('./experiments/results/0/fe_original_direct_'+g+'_'+str(ssize)+'_0_'+str(p)+'.txt',False,False)
                name,size,rsize,fitness1,fe_count1,time_count,sizes=readresults('./experiments/results/0/fe_model_direct_'+g+'_'+str(ssize)+'_0_'+str(p)+'.txt',True,False)
                name,size,rsize,fitness2,fe_count2,time_count,sizes=readresults('./experiments/results/0/fe_model_directwithK_'+g+'_'+str(ssize)+'_0_'+str(p)+'.txt',True,False)
                name,size,rsize,fitness3,fe_count3,time_count,sizes=readresults('./experiments/results/0/fe_model_dynamic_'+g+'_'+str(ssize)+'_0_'+str(p)+'.txt',False,True)
                plt.figure()
                plt.plot(fe_count0,fitness0,label='original')
                plt.plot(fe_count1,fitness1,label='model reduction')
                plt.plot(fe_count2,fitness2,label='model reduction with k')
                plt.plot(fe_count3,fitness3,label='dynamic reduction')
                plt.xlabel('fitness evaluation number')
                plt.ylabel('objective value')
                plt.legend()
                plt.title('the convergence curves on '+g+' instance with m='+str(ssize)+', p='+str(p))
                plt.savefig('./experiments/tem/figures/'+g+'_'+str(ssize)+'_p='+str(int(p*100)))
    