# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# I. COMPARAISON D'ALGORITHMES DE PLUS COURTS CHEMINS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Connaissance des algorithmes de plus courts chemins
# 1.1 Présentation de l’algorithme de Dijkstra
# -----------------------------------------------------------------------------

# → Exemple à faire à la main (petite matrice) et insérer ici la matrice et appel
M_dij = [
    [float('inf'), 2, 4, float('inf'), 1],
    [float('inf'), float('inf'), 3, 8, float('inf')],
    [float('inf'), float('inf'), float('inf'), 2, 6],
    [7, float('inf'), float('inf'), float('inf'), 5],
    [1, float('inf'), float('inf'), 4, float('inf')],
]

def Dijkstra(M, d):
    # Code identique
    ...

# -----------------------------------------------------------------------------
# 1.2 Présentation de l’algorithme de Bellman-Ford
# -----------------------------------------------------------------------------

# → Exemple à faire à la main avec poids négatifs (petite matrice)
M_bf = [
    [float('inf'), 6, float('inf'), 7, float('inf')],
    [float('inf'), float('inf'), 5, 8, -4],
    [float('inf'), -2, float('inf'), float('inf'), float('inf')],
    [float('inf'), float('inf'), -3, float('inf'), 9],
    [2, float('inf'), 7, float('inf'), float('inf')],
]

def Bellman_Ford(M, d):
    # Code identique
    ...

# -----------------------------------------------------------------------------
# 2. Dessin d’un graphe et d’un chemin à partir de sa matrice
# -----------------------------------------------------------------------------

# → Partie à compléter : utiliser des bibliothèques comme NetworkX ou Graphviz
# Exemple de fonction attendue :
# def dessiner_graphe(M): ...
# def dessiner_chemin(M, chemin): ...

# -----------------------------------------------------------------------------
# 3. Génération aléatoire de matrices de graphes pondérés
# -----------------------------------------------------------------------------

# 3.1 Graphe avec ~50% d’arcs
def graphe(n,a,b):
    ...

# 3.2 Graphe avec proportion variable p
def graphe2(n,p,a,b):
    ...

# Variante symétrique non orientée pour certains tests
def graphe3(n, p, a, b):
    ...

# -----------------------------------------------------------------------------
# 4. Codage des algorithmes de plus court chemin
# (déjà fait en 1.1 et 1.2)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. Influence du choix de la liste ordonnée des flèches
# -----------------------------------------------------------------------------

# Parcours largeur
def pl(M, s):
    ...

# Parcours profondeur
def pp(M, s):
    ...

# Variante de Bellman-Ford
def Bellman_Ford_variante(M, d, mode):
    ...

def test_variantes_BF():
    n = 50
    M = graphe3(n, 0.2, 1, 10)
    modes = ['aléatoire', 'pl', 'pp']
    for mode in modes:
        iters = Bellman_Ford_variante(M, 0, mode)
        print(f"Mode {mode} : {iters} itérations")

# -----------------------------------------------------------------------------
# 6. Comparaison expérimentale des complexités
# -----------------------------------------------------------------------------

# 6.1 Temps de calcul
def TempsDij(n):
    ...

def TempsBF(n):
    ...

# 6.2 Courbes et régression
valeurs = list(range(5, 201, 10))
temps_dij = [TempsDij(n) for n in valeurs]
temps_bf = [TempsBF(n) for n in valeurs]

plt.figure(figsize=(10, 6))
plt.plot(valeurs, temps_dij, label="Dijkstra", marker='o')
plt.plot(valeurs, temps_bf, label="Bellman-Ford (PL)", marker='s')
plt.xlabel("Taille du graphe (n)")
plt.ylabel("Temps (secondes)")
plt.title("Temps de calcul des plus courts chemins")
plt.legend()
plt.grid(True)
plt.show()

log_n = np.log(valeurs)
log_dij = np.log(np.array(temps_dij)+1e-8)
log_bf = np.log(np.array(temps_bf)+1e-8)

result_dij = linregress(log_n, log_dij)
result_bf = linregress(log_n, log_bf)

print(f"Exposant estimé (Dijkstra) a ≈ {result_dij.slope:.2f}")
print(f"Exposant estimé (Bellman-Ford PL) a ≈ {result_bf.slope:.2f}")

# 6.2 bis : cas p = 1/n
def TempsDij_variable(n):
    ...

def TempsBF_variable(n):
    ...

temps_dij_var = [TempsDij_variable(n) for n in valeurs]
temps_bf_var = [TempsBF_variable(n) for n in valeurs]

plt.figure(figsize=(10,6))
plt.plot(valeurs, temps_dij_var, label="Dijkstra (p=1/n)", marker='o')
plt.plot(valeurs, temps_bf_var, label="Bellman-Ford PL (p=1/n)", marker='s')
plt.xlabel("Taille du graphe (n)")
plt.ylabel("Temps (secondes)")
plt.title("Temps de calcul avec p = 1/n")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# II. SEUIL DE FORTE CONNEXITÉ D’UN GRAPHE ORIENTÉ
# =============================================================================

# -----------------------------------------------------------------------------
# 7. Test de forte connexité
# -----------------------------------------------------------------------------

def graphepourfc(n, p=0.5):
    ...

def Trans2(M):
    ...

def fc(M):
    ...

# -----------------------------------------------------------------------------
# 8. Forte connexité pour un graphe avec p=50%
# -----------------------------------------------------------------------------

def teststatfc(n, p=0.5):
    ...

def lafonctionquiaffiche(n, p=0.5):
    ...

# -----------------------------------------------------------------------------
# 9. Détermination du seuil de forte connexité
# -----------------------------------------------------------------------------

def seuil(n):
    ...

# Exemple
print(seuil(20))

# -----------------------------------------------------------------------------
# 10. Étude et identification de la fonction seuil
# -----------------------------------------------------------------------------

# 10.1 Graphique
def graphe_seuil(min_n=10, max_n=40):
    ...

graphe_seuil()

# 10.2 Régression log-log
def analyse_seuil_puissance(min_n=10, max_n=40):
    ...

analyse_seuil_puissance()
