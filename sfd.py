import random
from itertools import permutations,product


import pandas as pd
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

def dbscan_rysuj(data, liczba_sasiedzi=9,min_sample=8, show=False,plot3d=False):

    N_POINTS=liczba_sasiedzi
    nearest_neighbors = NearestNeighbors(n_neighbors=N_POINTS)
    neighbors = nearest_neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)
    distances = np.sort(distances[:,N_POINTS-1], axis=0)


    ##############   OPTYMALIZACJA epsilona   ##############

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

    #fig = plt.figure(figsize=(5, 5))
    #knee.plot_knee()
    #plt.xlabel("Points")
    #plt.ylabel("Distance")

    #print("Knee:",knee.knee)
    #print("Distances:",distances[knee.knee])

    optimal_eps = distances[knee.knee]

    #plt.show()


    ##############   DBSCAN   ##############
    dbscan_cluster = DBSCAN(eps=optimal_eps,min_samples=min_sample)
    dbscan_cluster.fit(data)

    # Visualizing DBSCAN
    if show==True:
        if plot3d==False:
            plt.scatter(data[:, 0],
            data[:, 1],
            c=dbscan_cluster.labels_)
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            txt = f'i={liczba_sasiedzi};j= {min_sample}'
            plt.text(0, -0.75, txt, fontsize=12)
            plt.show()

        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=dbscan_cluster.labels_)
            plt.show()
    return dbscan_cluster.labels_

def usuwanie(lista_klustry):
    for i in lista_klustry:
        if (-1 in lista_klustry):
            lista_klustry.remove(-1)
    lista_zw=lista_klustry
    return lista_zw

def kombinacje_df(labels,times):
    '''
    bierzemy liste odwiedzanych klustrow podajemy dlugosc ciagu, program tworzy wszystkie mozliwe ciagi o danej dlugosc
    z dostepnych klustrow i nastepnie zlicza jaki kluster wystapi po danym ciagu.
    koncowy wynik to tabela gdzie jako kolumny sa klustry a jako wiersze sa ciagi
    :param labels: lista odwiedzania klustrow w kolejnosci-list
    :param times: dlugosc ciagu-skalar
    :return: df: tabela typu dataframe
    '''
    new_labels=usuwanie(labels)
    dlugosc = len(set(new_labels))
    kombinacje = list(product(set(new_labels), repeat=times))
    kolumny=list(set(new_labels))
    df=pd.DataFrame(columns=kolumny,index=kombinacje)
    for i in kombinacje:
        for k in kolumny:
            suma=0
            for j in range(0, len(new_labels) - len(i)):
                if i==tuple(new_labels[j:j+len(i)]) and new_labels[j+len(i)]==k:
                    suma=suma+1
                df.iloc[kombinacje.index(i),kolumny.index(k)]=suma
    #dodawanie prawdopodobienstw do tabeli
    kolumny2=[]
    for i in kolumny:
        kolumny2.append(f'pr_next_{i}')
    for i in kolumny2:
        df[i]=None
    for i in range(len(kolumny2),2*len(kolumny2)):
        for j in range(len(kombinacje)):
            suma_pom=sum(df.iloc[j, 0:len(kolumny2)])
            if suma_pom==0:
                df.iloc[j,i]=0
            else:
                df.iloc[j,i]=df.iloc[j,i-len(kolumny2)]/suma_pom
    return df

def counter1(labels):
    counter={}
    for letter in labels:
        if letter not in counter:
            counter[letter] = 0
        counter[letter] += 1
    return counter

def max_dict(dict):
    maximum={'max':0,'kluster':0}
    for i,j in dict.items():
        if j>maximum['max']:
            maximum['max']=j
            maximum['kluster']=i
        continue
    return maximum

def usun_klustry(labels_klus,doc_ilosc_klus):
    '''
    chcemy pozbyc sie nie chcianych klustrow, podajemy liczbe:'ile chcesz klustrow'
    dalej program wybierza te najlicznieszne i gdy osiagnie zadana ilosc reszta jest kategoryzowana jakos halas (usuwana)
    :param labels_klus: lista odwiedzania klsutrow w kolejnosci-list
    :param doc_ilosc_klus: ile domyslnie powinnien uklad miec klustrow-skalar
    :return: labels: zwraca liste klustrow ale jedynie z tymi chcianymi, zachowujac kolejnosc naturalnie
    '''
    labels=usuwanie(labels_klus)
    counter = counter1(labels)
    empty = counter.copy()
    maxi = []
    i = 0
    for _ in counter.values():
        if i == doc_ilosc_klus:
            break
        maxi.append(max_dict(empty))
        del empty[max_dict(empty)['kluster']]
        i = i + 1

    new_list = []
    for i in maxi:
        new_list.append(i['kluster'])

    counter_copy = counter.copy()
    for i, j in counter.items():
        if (i in new_list):
            del counter_copy[i]
    # print(counter_copy)
    dlug = len(labels)
    for i, j in counter_copy.items():
        for _ in range(0, dlug):
            if i in labels:
                labels.remove(i)
    return labels

def lenlen(lista_2x):
    suma=0
    for i in lista_2x:
        suma=suma+len(i)
    return suma

def zlicz(lista_2x):
    mpty=[]
    for i in lista_2x:
        mpty.append([i[0],len(i)])
    return mpty

def licz_czasy(labels,halas=False):
    '''
    :param labels: lista odwiedzania klsutrow w kolejnosci
    :param halas: czy bierzemy pod uwage halas (domyslnie bierzemy pod uwage halas)
    :return: lista 'punktow' dwuwymiarowych, gdzie pierwsza wspolrzedna to kluster a druga to ile w nim przebywal
    '''
    if halas == True:
        labels=usuwanie(labels)
    mpty=[]
    for i in range(1,len(labels)):
        if labels[i-1] != labels[i]:
            mpty.append(labels[lenlen(mpty):i])
    return zlicz(mpty)

def entropia(pr_klustry):
    '''
    :param: pr_klustry- lista z prawdopodobienstwiem na wystapienie danego klustra
    :return: entropii - entropia dla logarytmu naturalnego
    '''
    v_pr_klustry = np.array(pr_klustry)
    v_pr = []
    for i in v_pr_klustry:
        v_pr.append(math.log(i, math.e))
    entropii=-sum(v_pr_klustry*v_pr)
    return entropii

def usuwanie_klustrow2(labels,procent=90,ignore_noise=False): # ewentualnie myslalem czy nie dodac jeszcze opcji ignorowania halasu
    dicti=counter1(labels)
    suma_sl=sum(dicti.values())
    suma_pom=0
    mpty=dicti.copy()
    for i,j in dicti.items():
        if suma_pom/suma_sl>procent*0.01:
            break
        else:
            suma_pom=suma_pom+max_dict(mpty)['max']
            del mpty[max_dict(mpty)['kluster']]
    for i in range(len(labels)):
        if labels[i] in list(mpty.keys()):
            labels[i]=-1

    return labels

def kombiancje_pr(labels,ciag_dl=5):
    mpty=[]
    for i in range(len(labels)-ciag_dl):
        mpty.append(tuple(labels[i:i+ciag_dl]))
    c=counter1(mpty)
    suma=sum(c.values())
    mpty1=[]
    for i in c.values():
        mpty1.append(i/suma)
    pass
    return mpty1

########################################
##############   gmcd   ##############
########################################

def f(a,epsilon,x):
    """Computation of the next vector in the GCM system (Kaneko 1990)."""
    fx = 1 - a * x * x
    return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

def data_help(N=50,epsilon=0.2,a=2.0,seed=5): # seed testowane 1, 5, ,10
    # the number of globally coupled maps
    #N = 50

    # the coupling term (should be between 0 and 1)
    #epsilon = 0.2

    # the parameter of the logistic map
    #a = 2.0

    # plotting time start and end
    t_start = 10000
    t_end = 20000

    # begin with pseudo-random initial condition
    #rad=random.randint(0,10)
    np.random.seed (seed)  #np.random.seed (rad)
    x = np.random.random ((N,))
    #print(x)
    # iterate the initial number of times
    for i in range(t_start):
        x = f(a,epsilon,x)

    # prepare an array for the points to plot
    data = np.empty ((t_end - t_start,3,))
    #print(data)
    # continue iterating and store the vectors in the array
    coord1,coord2,coord3 = 0,1,2
    for i in range(t_end - t_start):
        data[i,0] = x[coord1]
        data[i,1] = x[coord2]
        data[i,2] = x[coord3]
        x = f(a,epsilon,x)
    return data

def plot_hist(klustry, usuwanie=False):
    if usuwanie == True:
        klustry=usuwanie_klustrow2(klustry)
    mark=np.array(list(counter1(klustry).values()))
    sumka=sum(list(counter1(klustry).values()))
    marks=mark/sumka
    bars=set(klustry)
    y=np.arange(len(bars))
    plt.bar(y,marks,color='g')
    plt.xticks(y,bars)
    plt.show()
#################
# rysujemy




################################
########## testy własciwe
####################################
# style-['default','fivethirtyeight','seaborn', 'Solarize_Light2','classic']
plt.style.use('default')
#t1
data=data_help(N=50,epsilon=0.2,a=2.0)

klustry=dbscan_rysuj(data,6,min_sample=9,plot3d=True,show=True)
klustry1=klustry.copy()
kotki=usuwanie_klustrow2(klustry1)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=kotki)
plt.show()
aa=kombiancje_pr(kotki)
b=entropia(kombiancje_pr(usuwanie(list(klustry.copy())))) # bez hałasu
c=entropia(aa)
print(f'{c} ; {b}')
plot_hist(klustry)
plot_hist(kotki)

#t2

data=data_help(N=50,epsilon=0.2,a=1.95)
klustry=dbscan_rysuj(data,6,min_sample=9,plot3d=True,show=True)
klustry1=klustry.copy()
kotki=usuwanie_klustrow2(klustry1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=kotki)
plt.show()
aa=kombiancje_pr(kotki)
b=entropia(kombiancje_pr(usuwanie(list(klustry.copy()))))
c=entropia(aa)
print(f'{c} ; {b}')
plot_hist(klustry)
plot_hist(kotki)
#t3
data=data_help(N=50,epsilon=0.2,a=1.90)
klustry=dbscan_rysuj(data,6,min_sample=8,plot3d=True,show=True)

klustry1=klustry.copy()
kotki=usuwanie_klustrow2(klustry1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=kotki)
plt.show()
aa=kombiancje_pr(kotki)
b=entropia(kombiancje_pr(usuwanie(list(klustry.copy()))))
c=entropia(aa)
print(f'{c} ; {b}')
plot_hist(klustry)
plot_hist(kotki)
#t4
data=data_help(N=50,epsilon=0.19,a=2.0)
klustry=dbscan_rysuj(data,6,min_sample=8,plot3d=True,show=True)
klustry1=klustry.copy()
kotki=usuwanie_klustrow2(klustry1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=kotki)
plt.show()
aa=kombiancje_pr(kotki)
b=entropia(kombiancje_pr(usuwanie(list(klustry.copy()))))
c=entropia(aa)
print(f'{c} ; {b}')
plot_hist(klustry)
plot_hist(kotki)
#t5
data=data_help(N=50,epsilon=0.19,a=1.95)
klustry=dbscan_rysuj(data,6,min_sample=8,plot3d=True,show=True)

klustry1=klustry.copy()
kotki=usuwanie_klustrow2(klustry1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0],data[:, 1],data[:, 2],c=kotki)
plt.show()
aa=kombiancje_pr(kotki)
b=entropia(kombiancje_pr(usuwanie(list(klustry.copy()))))
c=entropia(aa)
print(f'{c} ; {b}')
plot_hist(klustry)
plot_hist(kotki)