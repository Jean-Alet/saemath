import numpy as np
import networkx as nx
import random
import time as t
import matplotlib.pyplot as plt


def Dijkstra(M,d):
    for i in M :
        return None


bfs = list(nx.bfs_tree(G, source='A'))
print("BFS à partir de A :", bfs)

#Parcours en profondeur (DFS),
dfs = list(nx.dfs_tree(G, source='A'))
print("DFS à partir de A :", dfs)



def graphe(n,a,b):
    matrice = [[0 for i in range(n)] for j in range(n)]
    if a >= b:
        a,b = b,a 
    for i in range(n):
        for j in range(n):
            matrice[i][j]= random.randint(0,1)
    for i in range(n):
        for j in range(n):
            if matrice[i][j] == 1:
                matrice[i][j] = random.randint(a, b)
            else:
                matrice[i][j] = float('inf')
    return matrice 

def graphe2(n,p,a,b):
    matrice = [[0 for i in range(n)] for j in range(n)]
    if a >= b:
        a,b = b,a 
    for i in range(n):
        for j in range(n):
            if random.random() < p:
                matrice[i][j] = random.randint(a, b)
            else:
                matrice[i][j] = float('inf')
    return matrice 

def graphe3(n, p, a, b):
    matrice = [[0 for i in range(n)] for j in range(n)]
    if a >= b:
        a, b = b, a
    for i in range(n):
        for j in range(n):
            matrice[i][j] = np.random.binomial(1, p)
    for i in range(n):
        for j in range(n):
            if matrice[i][j] == 1:
                matrice[i][j] = random.randint(a, b - 1)
            else:
                matrice[i][j] = float('inf')
    return matrice





n = 5
a = graphe(n,0,9)
print("[")
for i in range(n):
    print(a[i])
print("]")


def TempsDij(n):
    start = t.time()
    m = graphe3(0.2,n,0,9)
    m = Dijkstra(m,0)
    return (start-t.time())


def TempsBF(n):
    start = t.time()
    m = graphe3(0.2,n,0,9)
    m = BellmanFord(m,0)
    return (start-t.time())



valeurs = []
graph_Dij = []
graph_BF = []

for i in range(200):
    valeurs.append(i)
    graph_Dij.append(TempsDij(i))
    graph_BF.append(TempsBF(i))

plt.plot(valeurs, graph_Dij, marker='o', label='Dijkstra', color='blue')
plt.plot(valeurs, graph_BF, marker='s', label='Bellmanford', color='green')

plt.title("Comparaison des complexités")
plt.xlabel("Numéro grille")
plt.ylabel("Valeur complexité")
plt.legend()
plt.grid(False)

plt.show()

def Trans2(M):
    n=len(M)
    for i in range(n):
        for j in range(n):
            if M[j,i]==1:
                for k in range(n):
                    if M[i,k]==1:
                        M[j,k]=1
    return M

def fc(M):
    A = Trans2(M)
    n=len(M)
    for i in range(n):
        for j in range(n):
            if A[j,i]==1:
                return False
    return True

def teststatfc(n):
    pourcentage = 0 
    for i in range(500):
        if fc(graphe(n, 0, 10)):
            pourcentage +=1
    return pourcentage/5


for i in range(10) :
    print(teststatfc(i))