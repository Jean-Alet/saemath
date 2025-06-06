import numpy as np
import random
import time
import matplotlib.pyplot as plt

def graphe(n,a,b):
    matrice = [[float('inf') for i in range(n)] for j in range(n)]
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
    matrice = [[float('inf') for i in range(n)] for j in range(n)]
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
    matrice = [[float('inf') for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                matrice[i][j] = random.randint(a, b - 1)
                matrice[j][i] = matrice[i][j]
    return matrice

def Dijkstra(M, d):
    n = len(M)
    dist = [float('inf')] * n
    pred = [None] * n
    dist[d] = 0
    pred[d] = d

    non_visites = set(range(n))

    while len(non_visites) > 0:
        s = -1
        plus_petite_dist = float('inf')
        for i in non_visites:
            if dist[i] < plus_petite_dist:
                plus_petite_dist = dist[i]
                s = i

        if s == -1:
            non_visites.clear()
        else:
            non_visites.remove(s)

            for t in range(n):
                if M[s][t] != float('inf') and t in non_visites:
                    if dist[s] + M[s][t] < dist[t]:
                        dist[t] = dist[s] + M[s][t]
                        pred[t] = s

    resultats = {}
    for s in range(n):
        if s != d:
            if dist[s] == float('inf'):
                resultats[s] = "Sommet non joignable depuis " + str(d)
            else:
                chemin = []
                courant = s
                while courant != d:
                    chemin.append(courant)
                    courant = pred[courant]
                chemin.append(d)
                chemin.reverse()
                resultats[s] = (dist[s], chemin)

    return resultats 

def Bellman_Ford(M, d):
    n = len(M)
    dist = [float('inf')] * n
    pred = [None] * n

    dist[d] = 0
    pred[d] = d

    F = []
    for i in range(n):
        for j in range(n):
            if M[i][j] != float('inf'):
                F.append((i, j))

    modification = True
    iteration = 0

    while modification and iteration < n - 1:
        modification = False
        for (u, v) in F:
            if dist[u] + M[u][v] < dist[v]:
                dist[v] = dist[u] + M[u][v]
                pred[v] = u
                modification = True
        iteration += 1

    cycle_negatif = False
    idx = 0
    while not cycle_negatif and idx < len(F):
        u, v = F[idx]
        if dist[u] + M[u][v] < dist[v]:
            cycle_negatif = True
        idx += 1

    if cycle_negatif:
        print("Présence d'un cycle de poids négatif. Pas de plus court chemin fiable.")
        return None

    resultats = {}
    for s in range(n):
        if s != d:
            if dist[s] == float('inf'):
                resultats[s] = "Sommet non joignable depuis " + str(d)
            else:
                chemin = []
                courant = s
                while courant != d:
                    chemin.append(courant)
                    courant = pred[courant]
                chemin.append(d)
                chemin.reverse()
                resultats[s] = (dist[s], chemin)

    return resultats

def pl(M, s):
    n = len(M)
    couleur = {i: "blanc" for i in range(n)}
    couleur[s] = "vert"
    file = [s]
    Resultat = [s]
    while len(file) > 0:
        i = file[0]
        j = 0
        while j < n:
            if M[i][j] == 1 and couleur[j] == "blanc":
                file.append(j)
                couleur[j] = "vert"
                Resultat.append(j)
            j += 1
        file.pop(0)
    return Resultat

def pp(M, s):
    n = len(M)
    couleur = {i: "blanc" for i in range(n)}
    couleur[s] = "vert"
    pile = [s]
    Resultat = [s]
    while len(pile) > 0:
        i = pile[-1]
        Succ_blanc = []
        j = 0
        while j < n:
            if M[i][j] == 1 and couleur[j] == "blanc":
                Succ_blanc.append(j)
            j += 1
        if len(Succ_blanc) > 0:
            v = Succ_blanc[0]
            couleur[v] = "vert"
            pile.append(v)
            Resultat.append(v)
        else:
            pile.pop()
    return Resultat

def Bellman_Ford_variante(M, d, mode):
    n = len(M)
    dist = [float('inf')] * n
    pred = [None] * n
    dist[d] = 0
    pred[d] = d

    if mode == 'pl':
        ordre_sommets = pl(M, d)
    elif mode == 'pp':
        ordre_sommets = pp(M, d)
    else:
        ordre_sommets = list(range(n))
        random.shuffle(ordre_sommets)

    F = []
    for u in ordre_sommets:
        for v in range(n):
            if M[u][v] != float('inf'):
                F.append((u, v))

    modification = True
    iteration = 0
    while modification and iteration < n - 1:
        modification = False
        for (u, v) in F:
            if dist[u] + M[u][v] < dist[v]:
                dist[v] = dist[u] + M[u][v]
                pred[v] = u
                modification = True
        iteration += 1

    return iteration

def test_variantes_BF():
    n = 50
    M = graphe3(n, 0.2, 1, 10)
    modes = ['aléatoire', 'pl', 'pp']
    for mode in modes:
        iters = Bellman_Ford_variante(M, 0, mode)
        print(f"Mode {mode} : {iters} itérations")

def TempsDij(n):
    m = graphe3(n, 0.2, 1, 10)
    start = time.time()
    Dijkstra(m, 0)
    return time.time() - start

def TempsBF(n):
    m = graphe3(n, 0.2, 1, 10)
    start = time.time()
    Bellman_Ford_variante(m, 0, mode='pl')
    return time.time() - start

valeurs = list(range(5, 201, 10))
temps_dij = []
temps_bf = []

for n in valeurs:
    temps_dij.append(TempsDij(n))
    temps_bf.append(TempsBF(n))

plt.figure(figsize=(10, 6))
plt.plot(valeurs, temps_dij, label="Dijkstra", marker='o')
plt.plot(valeurs, temps_bf, label="Bellman-Ford (PL)", marker='s')
plt.xlabel("Taille du graphe (n)")
plt.ylabel("Temps (secondes)")
plt.title("Temps de calcul des plus courts chemins")
plt.legend()
plt.grid(True)
plt.show()

M_dij = [
    [float('inf'), 2, 4, float('inf'), 1],
    [float('inf'), float('inf'), 3, 8, float('inf')],
    [float('inf'), float('inf'), float('inf'), 2, 6],
    [7, float('inf'), float('inf'), float('inf'), 5],
    [1, float('inf'), float('inf'), 4, float('inf')],
]

M_bf = [
    [float('inf'), 6, float('inf'), 7, float('inf')],
    [float('inf'), float('inf'), 5, 8, -4],
    [float('inf'), -2, float('inf'), float('inf'), float('inf')],
    [float('inf'), float('inf'), -3, float('inf'), 9],
    [2, float('inf'), 7, float('inf'), float('inf')],
]

for i in range(len(M_dij)):
    print(M_dij[i])
for i in range(len(M_bf)):
    print(M_bf[i])
print(Dijkstra(M_dij, 1))
print(Bellman_Ford(M_bf, 1))

from scipy.stats import linregress

log_n = np.log(valeurs)
log_dij = np.log(np.array(temps_dij)+1e-8)
log_bf = np.log(np.array(temps_bf)+1e-8)

result_dij = linregress(log_n, log_dij)
result_bf = linregress(log_n, log_bf)

print(f"Exposant estimé (Dijkstra) a ≈ {result_dij.slope:.2f}")
print(f"Exposant estimé (Bellman-Ford PL) a ≈ {result_bf.slope:.2f}")

def TempsDij_variable(n):
    p = 1 / n
    m = graphe3(n, p, 1, 10)
    start = time.time()
    Dijkstra(m, 0)
    return time.time() - start

def TempsBF_variable(n):
    p = 1 / n
    m = graphe3(n, p, 1, 10)
    start = time.time()
    Bellman_Ford_variante(m, 0, mode='pl')
    return time.time() - start

temps_dij_var = [TempsDij_variable(n) for n in valeurs]
temps_bf_var = [TempsBF_variable(n) for n in valeurs]

plt.figure(figsize=(10,6))
plt.plot(valeurs, temps_dij_var, label="Dijkstra (p=1/n)", marker='o')
plt.plot(valeurs, temps_bf_var, label="Bellman-Ford PL (p=1/n)", marker='s')
plt.xlabel("Taille du graphe (n)")
plt.ylabel("Temps (secondes)")
plt.title("Temps de calcul des plus courts chemins avec p=1/n")
plt.legend()
plt.grid(True)
plt.show()

def graphepourfc(n, p=0.5):
    matrice = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() <= p:
                matrice[i][j] = 1
            else:
                matrice[i][j] = 0
    return matrice

def Trans2(M):
    n = len(M)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if M[i][k] and M[k][j]:
                    M[i][j] = 1
    return M

def fc(M):
    A = Trans2(M)
    n = len(M)
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                return False
    return True

def teststatfc(n, p=0.5):
    # valeur par défaut de p est a 0.5
    a = 200
    b = 0
    for i in range(a):
        if fc(graphepourfc(n, p)):
            b += 1
    return (100 * b / a)

def lafonctionquiaffiche(n, p=0.5):
    print("forte connexité avec proba de 1 :", p)
    for i in range(1, n+1):
        a = teststatfc(i, p)
        print("n = ",i,": ",a," %")

def seuil(n):
    p = 0.0
    while p <= 1.0:
        taux = teststatfc(n, p)
        if taux >= 99:
            return round(p, 2)
        p += 0.01
    return None

print(seuil(20))

import matplotlib.pyplot as plt

def graphe_seuil(min_n=10, max_n=40):
    X = []
    Y = []
    for n in range(min_n, max_n + 1):
        s = seuil(n)
        X.append(n)
        Y.append(s)
    
    plt.plot(X, Y, marker='o')
    plt.xlabel("Taille n du graphe")
    plt.ylabel("Seuil de forte connexité (p)")
    plt.title("Évolution du seuil de forte connexité en fonction de n")
    plt.grid(True)
    plt.show()
    
graphe_seuil()

def analyse_seuil_puissance(min_n=10, max_n=40):
    X = []
    Y = []
    for n in range(min_n, max_n + 1):
        s = seuil(n)
        if s is not None:
            X.append(n)
            Y.append(s)

    log_n = np.log(X)
    log_s = np.log(Y)
    pente, ordonnee_origine, _, _, _ = linregress(log_n, log_s)
    a = pente
    c = np.exp(ordonnee_origine)
    
    print(f"seuil(n) ≈ {c:.3f} × n^{a:.3f}")
    
    plt.plot(log_n, log_s, 'o', label="log(seuil)")
    plt.plot(log_n, a*log_n + ordonnee_origine, label=f"Régression : y = {a:.2f}x + {ordonnee_origine:.2f}")
    plt.xlabel("log(n)")
    plt.ylabel("log(seuil(n))")
    plt.title("Régression log-log de seuil(n)")
    plt.legend()
    plt.grid(True)
    plt.show()

analyse_seuil_puissance()