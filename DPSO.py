import numpy as np
import sys
import random
import time
import copy
from experiment import getFeatures

def improveIndivadual(x,indexes,lastindexes):
    a=np.zeros(np.max(indexes)+1)
    a[lastindexes]=x
    return a[indexes]

class Main:

    def __init__(self, f, c, goal_size,goal_fe,goal_time):
        self.best=float('inf')
        self.of, self.oc = np.array(f), np.array(c)
        self.m, self.n, self.f, self.c = len(f), len(c), np.array(f), np.array(c)
        self.goal_size = goal_size
        self.goal_fe,self.goal_time=goal_fe,goal_time
        self.generation,self.fe_count,self.time_count=0,0,0

    def DPSO(self,file):
        self.file=file
        self.time_count=time.process_time()
        self.initialize()
        self.generation=0
        file.write(f'{self.generation} {self.best} {self.fe_count} {(time.process_time()-self.time_count)}\n')
        while 1:
            self.generation += 1
            self.update_pbest_gbest()
            for individual in self.particle_list:
                individual.velocity()
                individual.cognition()
                individual.social(self.gbest_particle)
                individual.fitness=self.calculate_fitness(individual.solution)
                if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
                    break
            self.local_search()
            file.write(f'{self.generation} {self.best} {self.fe_count} {(time.process_time()-self.time_count)}\n')
            if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
                break
        file.flush()
        return self.gbest_particle.solution,self.best
    
    def DPSO_dynamic(self,file,model,a,b):
        self.file=file
        self.time_count=time.process_time()
        feature=getFeatures(self.of,self.oc)
        predict=model.predict_proba(feature)[:,1]
        indexes=np.argsort(-1*predict)
        k=self.findK(indexes)
        self.f=self.of[indexes[:k]]
        self.c=self.oc[:,indexes[:k]]
        self.m=len(self.f)
        self.n=len(self.c)
        self.initialize()
        self.generation=0
        file.write(f'{self.generation} {self.best} {self.fe_count} {(time.process_time()-self.time_count)} {k} {a} {b}\n')
        lastcount=0
        while 1:
            self.generation += 1
            last=self.gbest_particle.fitness
            self.update_pbest_gbest()
            if self.gbest_particle.fitness==last:
                lastcount+=1
            else:
                lastcount=0
            if lastcount>=a:
                lastcount=0
                if k!=len(self.of):
                    lastindexes=indexes[:k]
                    k=int(k+k/b)
                    if k>len(self.of):
                        k=len(self.of)
                    a+=1
                    indexes=np.argsort(-1*predict)[:k].tolist()
                    self.f=self.of[indexes[:k]]
                    self.c=self.oc[:,indexes[:k]]
                    self.m=len(self.f)
                    self.n=len(self.c)
                    for individual in self.particle_list:
                        individual.solution = improveIndivadual(individual.solution,indexes,lastindexes)
                        individual.pbest_solution = improveIndivadual(individual.pbest_solution,indexes,lastindexes)
                    self.gbest_particle.solution=improveIndivadual(self.gbest_particle.solution,indexes,lastindexes)
                    self.gbest_particle.pbest_solution=improveIndivadual(self.gbest_particle.pbest_solution,indexes,lastindexes)
            for individual in self.particle_list:
                individual.velocity()
                individual.cognition()
                individual.social(self.gbest_particle)
                individual.fitness=self.calculate_fitness(individual.solution)
                if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
                    break
            self.local_search()
            file.write(f'{self.generation} {self.best} {self.fe_count} {(time.process_time()-self.time_count)} {k} {a} {b}\n')
            if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
                break
        file.flush()
        return self.gbest_particle.solution,self.best
    
    def findK(self,index):
        left=0
        leftvalue=float('inf')
        right=len(index)+1
        rightvalue=float('inf')
        middle=int((left+right)/2)
        x=np.zeros(len(self.f))
        x[index[:middle]]=1
        middlevalue=self.calculate_fitness(x)
        while 1:
            if (right-left)==1:
                if leftvalue<rightvalue:
                    return left
                else:
                    return right
            leftmiddle=int((left+middle)/2)
            x=np.zeros(len(self.f))
            x[index[:leftmiddle]]=1
            leftmiddlevalue=self.calculate_fitness(x)
            if leftmiddlevalue<middlevalue:
                right=middle
                rightvalue=middlevalue
                middle=leftmiddle
                middlevalue=leftmiddlevalue
                continue
            rightmiddle=int((middle+right)/2)
            x=np.zeros(len(self.f))
            x[index[:rightmiddle]]=1
            rightmiddlevalue=self.calculate_fitness(x)
            if rightmiddlevalue<middlevalue:
                left=middle
                leftvalue=middlevalue
                middle=rightmiddle
                middlevalue=rightmiddle
            else:
                left=leftmiddle
                leftvalue=leftmiddlevalue
                right=rightmiddle
                rightvalue=rightmiddlevalue

    def initialize(self):
        self.particle_list = []
        self.gbest_particle = particle(np.zeros(self.m),int(sys.maxsize))
        for j in range(0, self.goal_size):
            x=np.random.randint(0, 2, self.m)
            fitness=self.calculate_fitness(x)
            individual = particle(x,fitness)
            self.particle_list.append(individual)
            if individual.fitness < self.gbest_particle.fitness:
                self.gbest_particle = individual
        self.gbest_particle = copy.deepcopy(self.gbest_particle)
    
    def calculate_fitness(self,facility_vector):
        result = np.inner(facility_vector, self.f)
        open_facility_indexes = np.where(facility_vector)[0]
        if len(open_facility_indexes) == 0:
            result = sys.maxsize
            return result
        temp = np.min(self.c[:, open_facility_indexes], axis=1)
        result += np.sum(temp)
        if result<self.best:
            self.best=result
        self.fe_count+=1
        if self.fe_count%100==0:
            self.file.write(f'{self.generation} {self.best} {self.fe_count} {(time.process_time()-self.time_count)}\n')
        return result

    def update_pbest_gbest(self):
        for individual in self.particle_list:
            if individual.fitness < self.gbest_particle.fitness:
                self.gbest_particle = individual
            if individual.fitness < individual.pbest_fitness:
                individual.pbest_fitness = individual.fitness
                individual.pbest_solution = individual.solution
        self.gbest_particle = copy.deepcopy(self.gbest_particle)

    def local_search(self):
        s0 = np.copy(self.gbest_particle.solution)
        index1, index2 = np.random.randint(0,self.m,2)
        while index1==index2:
            index1, index2 = np.random.randint(0,self.m,2)
        s0[index1] = 1-s0[index1]
        s0[index2] = 1-s0[index2]
        s = s0
        if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
            return
        fs = self.calculate_fitness(s)
        for l in range(0, self.m):
            rand_facility_loc = random.randint(0, self.m-1)
            s1 = np.copy(s)
            s1[rand_facility_loc] = 1-s1[rand_facility_loc]
            if ((self.goal_fe!=-1 and self.fe_count>=self.goal_fe) or (self.goal_time!=-1 and time.process_time()-self.time_count>=self.goal_time)):
                return
            fs1 = self.calculate_fitness(s1)
            if fs1 < fs:
                s = s1
                fs = fs1
        if fs <= self.gbest_particle.fitness:
            self.gbest_particle.solution = s
            self.gbest_particle.fitness = fs


class particle:
    c1 = 0.5  # social probability
    c2 = 0.5  # cognitive probability
    w = 0.9  # inertia weight

    def __init__(self,x,fitness):
        self.m=len(x)
        self.solution = x
        self.pbest_solution = x
        self.fitness = fitness
        self.pbest_fitness = fitness

    def velocity(self):
        r = random.random()
        if r < particle.w:
            index1, index2 = np.random.randint(0,self.m,2)
            while index1==index2:
                index1, index2 = np.random.randint(0,self.m,2)
            a = self.solution[index1]
            self.solution[index1] = self.solution[index2]
            self.solution[index2]=a

    def cognition(self):
        r = random.random()
        if r < particle.c1:
            index = np.random.randint(1,self.m)
            if random.random()<0.5:
                self.solution[index:]=self.pbest_solution[index:]
            else:
                self.solution[:index]=self.pbest_solution[:index]

    def social(self,gbest_particle):
        r = random.random()
        if r < particle.c2:
            index1 = np.random.randint(0,self.m-1)
            index2 = np.random.randint(index1+1,self.m)
            if random.random()<0.5:
                self.solution[index1:index2]=gbest_particle.solution[index1:index2]
            else:
                self.solution[:index1]=gbest_particle.solution[:index1]
                self.solution[index2:]=gbest_particle.solution[index2:]