import random
import numpy as np
from docplex.mp.model import Model
import sys

def functionValue(f,c,y):
    result = np.inner(y, f)
    open_facility_indexes = np.where(y)[0]
    if len(open_facility_indexes) == 0:
        result = sys.maxsize
        return result
    temp = np.min(c[:, open_facility_indexes], axis=1)
    result += np.sum(temp)
    return result

def generateRandom(mRange,isE):
    # n/m from 1 to 10
    m=random.randint(mRange[0],mRange[1])
    n=m*random.randint(1,10)
    #c from 1 to 200, f/c from 1 to 20
    maxf=200*random.randint(1,20)
    f=np.random.rand(m,)*(maxf-10)+10
    if(isE):
        c=np.zeros((n,m))
        #square with size(141，141)，the largest distance is 141*1.414=200
        mlocation=np.random.rand(m,2)*141
        nlocation=np.random.rand(n,2)*141
        for i in range(n):
            for j in range(m):
                c[i][j]=np.sqrt((nlocation[i][0]-mlocation[j][0])*(nlocation[i][0]-mlocation[j][0])+(nlocation[i][1]-mlocation[j][1])*(nlocation[i][1]-mlocation[j][1]))
        return f,c
    else:
        c=np.random.rand(n,m)*199+1
        return f,c

def generateRandomTest(m,n,f_min,f_max):
    f=np.random.rand(m,)*(f_max-f_min)+f_min
    c=np.zeros((n,m))
    #square with size(141，141)，the largest distance is 141*1.414=200
    mlocation=np.random.rand(m,2)*141
    nlocation=np.random.rand(n,2)*141
    for i in range(n):
        for j in range(m):
            c[i][j]=np.sqrt((nlocation[i][0]-mlocation[j][0])*(nlocation[i][0]-mlocation[j][0])+(nlocation[i][1]-mlocation[j][1])*(nlocation[i][1]-mlocation[j][1]))
    return f,c

def generateOR(m,n,p):
    mlocation=np.random.rand(m,2)*1000
    nlocation=np.random.rand(n,2)*1000
    alpha=np.random.rand()*0.25+1
    d=np.random.rand(n)*99+1
    c=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            c[i][j]=np.sqrt((nlocation[i][0]-mlocation[j][0])*(nlocation[i][0]-mlocation[j][0])+(nlocation[i][1]-mlocation[j][1])*(nlocation[i][1]-mlocation[j][1]))
            c[i][j]*=alpha*d[i]
    set1=set()
    while set1.__len__()<p+1:
        set1.add(np.random.randint(m))
    x1=np.zeros(m)
    x1[list(set1)]=1
    set1.remove(random.choice(list(set1)))
    set1.remove(random.choice(list(set1)))
    x2=np.zeros(m)
    x2[list(set1)]=1
    f=(np.random.rand(m)*0.5+0.75)*(functionValue(np.zeros(m),c,x2)-functionValue(np.zeros(m),c,x1))/2
    f=np.round(f)
    c=np.round(c,5)
    return f,c,d

def generateM(m,n,fmin,fmax,cmin,cmax,bmin,bmax):
    b=np.random.random(n)*(bmax-bmin)+bmin
    c=np.zeros((n,m))
    for i in range(n):
        c[i]=np.random.random(m)*(cmax-cmin)+cmin
        c[i]=c[i]*b[i]
    s=np.sum(c.T,axis=1)
    smax=np.max(s)
    smin=np.min(s)
    f=fmax-(s-smin)*(fmax-fmin)/(smax-smin)
    f=np.round(f,3)
    c=np.round(c,5)
    return f,c,b

def generateGap(m,f,cinf,l,kind):
    n=m
    f=np.ones(m)*f
    c=np.ones((n,m))*cinf
    column=np.zeros(n)
    setI=set()
    for i in range(n):
        setI.add(i)
    for j in range(m):
        l0=0
        if kind=='C':
            for i in range(n):
                if m-j==l-column[i]:
                    l0+=1
                    column[i]+=1
                    if i in setI:
                        setI.remove(i)
                    c[i][j]=np.random.randint(5)
        setII=setI.copy()
        while setII.__len__()>l-l0:
            setII.remove(random.choice(list(setII)))
        for i in list(setII):
            c[i][j]=np.random.randint(5)
            column[i]+=1
            if kind=='C':
                if column[i]==l:
                    setI.remove(i)
    if kind=='A':
        c=c.T
    return f,c

def writeRandom(f,c,label,value,filename,solname):
    file=open(filename,'w')
    file.write(str(len(f))+' '+str(len(c))+'\n')
    for ff in f:
        file.write(str(ff)+' ')
    file.write('\n')
    for i in range(len(c)):
        for j in range(len(f)):
            file.write(str(c[i][j])+' ')
        file.write('\n')
    file.close()
    if label!=None:
        file=open(solname,'w')
        for x in label:
            file.write(str(x)+' ')
        file.write(str(value)+'\n')
        file.close()

def writeOR(f,c,d,label,value,filename,solname):
    file=open(filename,'w')
    file.write(' '+str(len(f))+' '+str(len(c))+' \n')
    for ff in f:
        file.write(' capacity '+str(ff)+' \n')
    for i in range(len(c)):
        file.write(str(d[i])+'\n')
        count=0
        for j in range(len(f)):
            count+=1
            if count==1:
                file.write(' '+str(c[i][j])+' ')
            elif count==7:
                file.write(str(c[i][j])+' \n')
                count=0
            else:
                file.write(str(c[i][j])+' ')
        if count!=0:
            file.write('\n')
    file.close()
    if label!=None:
        file=open(solname,'w')
        for x in label:
            file.write(str(x)+' ')
        file.write(str(value)+'\n')
        file.close()

def writeM(f,c,b,label,value,filename,solname):
    file=open(filename,'w')
    file.write(str(len(f))+' '+str(len(c))+'\n')
    for ff in f:
        file.write('0 '+str(round(ff,3))+'\n')
    for i in range(len(c)):
        file.write(str(b[i])+'\n')
        for j in range(len(f)):
            file.write(str(round(c[i][j],5))+' ')
        file.write('\n')
    file.close()
    if label!=None:
        file=open(solname,'w')
        for x in label:
            file.write(str(x)+' ')
        file.write(str(value)+'\n')
        file.close()

def writeGap(f,c,label,value,filename,solname):
    file=open(filename,'w')
    file.write(str(len(f))+' '+str(len(c))+' 0 \n')
    for j in range(len(f)):
        file.write(str(j+1)+' '+str(f[j])+' ')
        for i in range(len(c)):
            file.write(str(c[i][j])+' ')
        file.write('\n')
    file.close()
    if label!=None:
        file=open(solname,'w')
        for x in label:
            file.write(str(x)+' ')
        file.write(str(value)+'\n')
        file.close()

# read instances from Random dataset
def readRandom(filepath,solpath):
    file=open(filepath)
    s=file.readline()
    ss=s.replace('\n','').split(' ')
    m=eval(ss[0])
    n=eval(ss[1])
    f=[]
    s=file.readline()
    ss=s.replace('\n','').split(' ')
    for j in range(m):
        f.append(eval(ss[j]))
    c=[[] for i in range(n)]
    for i in range(n):
        s=file.readline()
        ss=s.replace('\n','').split(' ')
        for j in range(m):
            c[i].append(eval(ss[j]))
    file.close()
    label=None
    if solpath is not None:
        file=open(solpath)
        label=[]
        s=file.readline()
        ss=s.replace('\n','').split(' ')
        for j in range(m):
            label.append(np.round(eval(ss[j])))
        file.close()
    return f,c,label

# read instances from OR dataset
def readOR(filepath,solpath):
    file=open(filepath)
    s=file.readline().split(" ")
    m,n=eval(s[1]),eval(s[2])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().split(" ")
        f[j]=eval(s[2])
    for i in range(n):
        s=file.readline()
        count=0
        while count<m:
            s=file.readline().split(" ")
            for ss in s:
                if (ss=='') or (ss=='\n'):
                    continue
                c[i][count]=eval(ss)
                count+=1
    file.close()
    label=None
    if solpath is not None:
        file=open(solpath)
        label=[]
        s=file.readline()
        ss=s.replace('\n','').split(' ')
        for j in range(m):
            label.append(np.round(eval(ss[j])))
        file.close()
    return f,c,label

# read instances from M* dataset
def readM(filepath,solpath):
    file=open(filepath)
    s=file.readline().split(" ")
    m,n=eval(s[0]),eval(s[1])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().split(" ")
        f[j]=eval(s[1])
    for i in range(n):
        s=file.readline()
        s=file.readline().split(" ")
        for j in range(m):
            c[i][j]=eval(s[j])
    file.close()
    label=None
    if solpath is not None:
        file=open(solpath)
        label=[]
        s=file.readline()
        ss=s.replace('\n','').split(' ')
        for j in range(m):
            label.append(np.round(eval(ss[j])))
        file.close()
    return f,c,label

# read instances from Euclid,GapA,GapB,GapC dataset
def readGap(filepath,solpath,emptyLine):
    file=open(filepath)
    if emptyLine:
        s=file.readline()
    s=file.readline().split(" ")
    m,n=eval(s[0]),eval(s[1])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().replace('\n','').split(" ")
        f[j]=eval(s[1])
        for i in range(n):
            c[i][j]=s[i+2]
    file.close()
    label=None
    if solpath is not None:
        file=open(solpath)
        label=[]
        s=file.readline()
        ss=s.replace('\n','').split(' ')
        for j in range(m):
            label.append(np.round(eval(ss[j])))
        file.close()
    return f,c,label

def getLabel(f,c):
    solution,value=solveByCplex(f,c)
    return solution,value

def solveByCplex(f,c):
    # 定义模型种类， 这里是混合整数规划“MIP"
    mip = Model('MIP') #mdl是英文单词“model" 的缩写
    x = mip.binary_var_dict(len(f), name='x')
    y = {(i,j): mip.binary_var(name="y_{0}_{1}".format(i,j)) for i in range(len(c)) for j in range(len(f))}
    mip.minimize(mip.sum(f[j]*x[j] for j in range(len(f)))+mip.sum(c[i][j]*y[i,j] for i in range(len(c)) for j in range(len(f))))
    for i in range(len(c)):
        mip.add_constraint(mip.sum(y[i,j] for j in range(len(f)))==1)
    for i in range(len(c)):
        for j in range(len(f)):
            mip.add_constraint(y[i,j]<=x[j])
    solution = mip.solve()
    x=solution._get_all_values()[:len(f)]
    return x,solution._objective
