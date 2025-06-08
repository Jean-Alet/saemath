# =============================================================================
# 
# 
# 
# =============================================================================

from graphviz import Digraph
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
# 1.1 Présentation de l'algorithme de  Dijkstra
# -----------------------------------------------------------------------------

# Exemple à faire à la main (petite matrice) et insérer ici la matrice et appel
M_dij = [
    [float('inf'), 2, 4, float('inf'), 1],
    [float('inf'), float('inf'), 3, 8, float('inf')],
    [float('inf'), float('inf'), float('inf'), 2, 6],
    [7, float('inf'), float('inf'), float('inf'), 5],
    [1, float('inf'), float('inf'), 4, float('inf')],
]

# -----------------------------------------------------------------------------
# 1.2 Présentation de l’algorithme de Bellman-Ford
# -----------------------------------------------------------------------------

# Exemple à faire à la main avec poids négatifs (petite matrice)
M_bf = [
    [float('inf'), 6, float('inf'), 7, float('inf')],
    [float('inf'), float('inf'), 5, 8, -4],
    [float('inf'), -2, float('inf'), float('inf'), float('inf')],
    [float('inf'), float('inf'), -3, float('inf'), 9],
    [2, float('inf'), 7, float('inf'), float('inf')],
]

# -----------------------------------------------------------------------------
# 2. Dessin d’un graphe et d’un chemin à partir de sa matrice
# -----------------------------------------------------------------------------

def afficher_graphe_oriente(matrice, chemin=None, nom_fichier='graphe_oriente'):
    """
    Affiche un graphe orienté pondéré à partir d'une matrice d'adjacence, 
    en mettant en évidence un chemin particulier s'il est fourni.

    Le graphe est généré au format PNG à l'aide de la bibliothèque Graphviz.

    Args:
        matrice (list[list[float]]): Matrice d'adjacence représentant les poids des arêtes. 
            Une valeur de 0 ou np.inf indique l'absence d arête.
        chemin (list[int], optional): Liste de sommets représentant un chemin à surligner en rouge. 
            Par défaut, aucun chemin n est surligné.
        nom_fichier (str, optional): Nom du fichier de sortie sans extension. 
            Par défaut, 'graphe_oriente'.

    Returns:
        None: La fonction génère un fichier image PNG et l'ouvre automatiquement dans le visualiseur par défaut.
    """

    if chemin is None:
        chemin = []

    aretes_chemin = list(zip(chemin, chemin[1:]))

    dot = Digraph(format='png')
    dot.attr(rankdir='LR')  # Orientation gauche à droite

    # Ajout des noeuds avec coloration spéciale pour ceux présents dans le chemin
    for i in range(len(matrice)):
        label = str(i)
        color = 'lightcoral' if i in chemin else 'lightblue'
        dot.node(label, style='filled', fillcolor=color)

    # Ajout des arêtes avec coloration spéciale pour celles du chemin
    for i in range(len(matrice)):
        for j in range(len(matrice[i])):
            poids = matrice[i][j]
            if poids != 0 and not np.isinf(poids):
                couleur = 'red' if (i, j) in aretes_chemin else 'black'
                dot.edge(str(i), str(j), label=str(int(poids)), color=couleur)

    dot.render(nom_fichier, view=True)

# -----------------------------------------------------------------------------
# 3. Génération aléatoire de matrices de graphes pondérés
# -----------------------------------------------------------------------------

# 3.1 Graphe avec ~50% d’arcs

def graphe(n,a,b):
    """
    Génère une matrice d'adjacence représentant un graphe orienté 
    avec environ 50% d'arcs, pondérés aléatoirement entre les valeurs a et b.

    Args:
        n (int): Nombre de sommets du graphe.
        a (int): Borne inférieure des poids des arêtes (incluse).
        b (int): Borne supérieure des poids des arêtes (incluse).

    Returns:
        list[list[float]]: Matrice d'adjacence du graphe, où une valeur 
        float('inf') signifie l'absence d'arête entre deux sommets.
    """
    matrice = [[float('inf') for i in range(n)] for j in range(n)]

    if a >= b:  # Pour vérifier que a est inférieur ou égal à b
        a,b = b,a 

    # Placement aléatoire de 0 ou 1 entre les sommets
    for i in range(n): 
        for j in range(n):
            matrice[i][j]= random.randint(0,1) # Environ 50% de chances

    # Attribution des poids
    for i in range(n):
        for j in range(n):
            if i==j:
                matrice[i][j] = float('inf') # Pour supprimer les boucles
            if matrice[i][j] == 1:
                matrice[i][j] = random.randint(a, b) # Poids entre a et b pour les autres points
            else:
                matrice[i][j] = float('inf')
    return matrice

# 3.2 Graphe avec proportion variable p

def graphe2(n,p,a,b):
    """
    Génère une matrice d'adjacence représentant un graphe orienté pondéré,
    où chaque arrête a une probabilité p d'exister et un poids aléatoire 
    compris entre a et b.

    Args:
        n (int): Nombre de sommets du graphe.
        p (float): Probabilité qu'un arc existe entre deux sommets (0 ≤ p ≤ 1).
        a (int): Borne inférieure des poids des arêtes (incluse).
        b (int): Borne supérieure des poids des arêtes (incluse).

    Returns:
        list[list[float]]: Matrice d'adjacence du graphe, où float('inf') signifie
        l'absence d'arête entre deux sommets.
    """
    # Initialisation de la matrice avec des valeurs infinies 
    matrice = [[float('inf') for i in range(n)] for j in range(n)]
    
    if a >= b: # Pour vérifier que a est inférieur ou égal à b
        a,b = b,a 

    for i in range(n):
        for j in range(n):
            if i==j:
                matrice[i][j] = float('inf') # Pas de boucle
            if random.random() < p: # Mettre une proba p d'arrêtes
                matrice[i][j] = random.randint(a, b) # Ajout d'un arc avec poids entre a et b
            else:
                matrice[i][j] = float('inf')
    return matrice

# Variante symétrique non orientée pour certains tests
def graphe3(n, p, a, b):
   # -----------------------IMPORTANT JE NE SUIS PAS SUR QUE LA FONCTION SOIT CORRECTE ---------------------------------
    # Initialisation d'une matrice n x n 
    matrice = [[float('inf') for i in range(n)] for j in range(n)]
    
    # Construction uniquement sur la moitié supérieure pour éviter les doublons
    for i in range(n):
        for j in range(i + 1, n):
            if i==j:
                matrice[i][j] = float('inf')
            if random.random() < p:
                matrice[i][j] = random.randint(a, b - 1)
                matrice[j][i] = matrice[i][j]
    return matrice




# -----------------------------------------------------------------------------
# 4. Codage des algorithmes de plus court chemin
# (déjà fait en 1.1 et 1.2)
# -----------------------------------------------------------------------------

def Dijkstra(M, d):
    """
    Implémente l'algorithme de Dijkstra pour trouver les plus courts chemins
    depuis un sommet source d vers tous les autres sommets dans un graphe pondéré.

    Args:
        M (list[list[float]]): Matrice d'adjacence représentant le graphe. 
            Une valeur float('inf') signifie l'absence d'arête entre deux sommets.
        d (int): Indice du sommet de départ.

    Returns:
        dict: Dictionnaire où chaque clé est un sommet de destination `s` et chaque valeur est :
              - un tuple (distance, chemin) si `s` est atteignable depuis `d`
              - une chaîne indiquant que le sommet n'est pas joignable sinon.
    """
    n = len(M)                       # Nombre de sommets
    dist = [float('inf')] * n        # Distances minimales depuis le sommet source
    pred = [None] * n                # Prédécesseurs 
    dist[d] = 0                      # Distance du sommet source à lui-même
    pred[d] = d                      # Le prédécesseur du sommet source est lui-même

    non_visites = set(range(n))      # Ensemble des sommets pas visités

    while len(non_visites) > 0:
        s = -1
        plus_petite_dist = float('inf')
        for i in non_visites:
            if dist[i] < plus_petite_dist:
                plus_petite_dist = dist[i]
                s = i

        if s == -1: # Aucun sommet accessible restant 
            non_visites.clear()
        else:
            non_visites.remove(s)

            # Mise à jour des distances pour les voisins de s
            for t in range(n):
                if M[s][t] != float('inf') and t in non_visites:
                    if dist[s] + M[s][t] < dist[t]:
                        dist[t] = dist[s] + M[s][t]
                        pred[t] = s

    # Résultats correspond aux distances et chemins depuis la source
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
    """
    Applique l'algorithme de Bellman-Ford pour trouver les plus courts chemins 
    depuis un sommet source `d` dans un graphe pondéré pouvant contenir des poids négatifs.

    Args:
        M (list[list[float]]): Matrice d'adjacence du graphe. 
            float('inf') indique l'absence darête entre deux sommets.
        d (int): Indice du sommet source.

    Returns:
        dict or None: 
            - Si aucun cycle négatif nest détecté, retourne un dictionnaire où :
                * chaque clé est un sommet atteignable depuis `d`
                * chaque valeur est un tuple (distance, chemin) ou un message pour les sommets inaccessibles
            - Si un cycle de poids négatif est détecté, affiche un message et retourne None.
    """

    n = len(M)
    dist = [float('inf')] * n       # Distances minimales depuis la source
    pred = [None] * n               # Prédécesseurs

    dist[d] = 0                     # La distance du sommet source à lui-même est 0
    pred[d] = d                     # Le prédécesseur du sommet source est lui-même

    F = []                          # Liste des arêtes (u, v) du graphe

    for i in range(n):
        for j in range(n):
            if M[i][j] != float('inf'):
                F.append((i, j))    # Ajout des arêtes valides

    modification = True             # Indique si une mise à jour a été faite lors de l'itération
    iteration = 0

    # Les distances sont mises à jour à partir des arêtes, jusqu'à stabilisation ou après n - 1 itérations
    while modification and iteration < n - 1:
        modification = False
        for (u, v) in F:
            if dist[u] + M[u][v] < dist[v]:
                dist[v] = dist[u] + M[u][v]
                pred[v] = u
                modification = True
        iteration += 1

    # Détection de cycle de poids négatif
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


    # Construction des chemins et des résultats
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


#resultats = Dijkstra(M_dij, 0)
#print(resultats)
#(a, b) = resultats[3]
#afficher_graphe_oriente(M_dij, b, nom_fichier='dijkstra')

#resultats = Bellman_Ford(M_bf, 0)
#print(resultats)
#(a, b) = resultats[4]
#afficher_graphe_oriente(M_bf, b, nom_fichier='bellman_ford')

#a = graphe(7,0,10)
#a = graphe3(7,-3,10,0.3)
#resultats = Dijkstra(M_dij, 0)
#print(resultats)
#(a, b) = resultats[3]
#afficher_graphe_oriente(M_dij, b, nom_fichier='dijkstra')
#resultats2 = Bellman_Ford(M_bf, 0)
#print(resultats2)
#(a, b) = resultats2[4]
#afficher_graphe_oriente(M_bf, b, nom_fichier='bellman_ford')

# -----------------------------------------------------------------------------
# 5. Influence du choix de la liste ordonnée des flèches
# -----------------------------------------------------------------------------

# Parcours largeur
def pl(M, s):
    """
    Effectue un parcours en largeur (BFS) d un graphe à partir d un sommet source.

    Args:
        M (list[list[int]]): Matrice d adjacence du graphe (valeurs 1 pour les arêtes, 0 sinon).
        s (int): Indice du sommet de départ.

    Returns:
        list[int]: Liste des sommets visités dans l ordre du parcours en largeur.
    """
    n = len(M)
    couleur = {i: "blanc" for i in range(n)}
    couleur[s] = "vert"
    file = [s]                                 # File pour gérer les sommets à explorer
    Resultat = [s]                             # Liste des sommets visités
    while len(file) > 0:
        i = file[0]                             # Sommet en tête de file
        j = 0
        while j < n:
            if M[i][j] != float('inf') and couleur[j] == "blanc":
                file.append(j)                 # On ajoute le voisin non visité
                couleur[j] = "vert"            # Marqué comme visité
                Resultat.append(j)
            j += 1
        file.pop(0)                            # On retire le sommet traité de la file
    return Resultat

# Parcours profondeur
def pp(M, s):
    """
    Effectue un parcours en profondeur (DFS) d un graphe à partir d un sommet source.

    Args:
        M (list[list[int]]): Matrice d adjacence du graphe (valeurs 1 pour les arêtes, 0 sinon).
        s (int): Indice du sommet de départ.

    Returns:
        list[int]: Liste des sommets visités dans l ordre du parcours en profondeur.
    """
    n = len(M)
    couleur = {i: "blanc" for i in range(n)}
    couleur[s] = "vert"
    pile = [s]                                # Pile pour la stratégie de parcours en profondeur
    Resultat = [s]
    while len(pile) > 0:
        i = pile[-1]                           # Sommet au sommet de la pile
        Succ_blanc = []                        # Liste des successeurs non visités
        j = 0
        while j < n:
            if M[i][j] != float('inf') and couleur[j] == "blanc":
                Succ_blanc.append(j)
            j += 1
        if len(Succ_blanc) > 0:
            v = Succ_blanc[0]
            couleur[v] = "vert"
            pile.append(v)
            Resultat.append(v)
        else:
            pile.pop()                         # Aucun successeur non visité
    return Resultat

# Variante de Bellman-Ford
def Bellman_Ford_variante(M, d, mode='Aléatoire'):
    """
    Variante de l'algorithme de Bellman-Ford utilisant un ordre de parcours spécifique des sommets.

    Cette fonction applique l'algorithme de Bellman-Ford en explorant les arêtes selon un ordre déterminé
    par un parcours en largeur, en profondeur, ou un ordre aléatoire des sommets.

    Args:
        M (list[list[float]]): Matrice d'adjacence du graphe pondéré.
            Les absences d'arêtes doivent être codées avec float('inf').
        d (int): Indice du sommet source.
        mode (str, optional): Mode de parcours des sommets :
            - 'pl' : parcours en largeur (BFS)
            - 'pp' : parcours en profondeur (DFS)
            - tout autre valeur : ordre aléatoire
            Par défaut, le mode est 'Aléatoire'.

    Returns:
        tuple:
            - str: Message indiquant le nombre d itérations effectuées dans la variante.
            - dict or None: Résultat de l appel à Bellman_Ford classique depuis le même sommet.
                Le dictionnaire associe à chaque sommet atteint une paire (distance, chemin),
                ou une chaîne indiquant l'inaccessibilité. Retourne None en cas de cycle négatif.
    """
    n = len(M)
    dist = [float('inf')] * n
    pred = [None] * n
    dist[d] = 0
    pred[d] = d

    # Détermination de l’ordre des sommets selon le mode choisi
    if mode == 'pl':
        ordre_sommets = pl(M, d)
    elif mode == 'pp':
        ordre_sommets = pp(M, d)
    else:
        ordre_sommets = list(range(n))
        random.shuffle(ordre_sommets)

    # Construction de la liste des arêtes selon cet ordre
    F = []
    for u in ordre_sommets:
        for v in range(n):
            if M[u][v] != float('inf'):
                F.append((u, v))

    # Mise à jour des distances 
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
        message = "Itération : " + str(iteration)

    return (message, Bellman_Ford(M, d))


def test_variantes_BF():
    """
    Teste les différentes variantes de l'algorithme de Bellman-Ford sur un graphe généré.

    Génère un graphe non orienté aléatoire avec 5 sommets et une densité de 20 % d'arêtes.
    Applique la variante de Bellman-Ford avec trois ordres de parcours :
        - 'aléatoire' : ordre aléatoire des sommets
        - 'pl' : parcours en largeur
        - 'pp' : parcours en profondeur

    Affiche le nombre d'itérations nécessaires pour chaque mode.
    """
    n = 5
    M = graphe3(n, 0.2, 1, 10) 
    modes = ['aléatoire', 'pl', 'pp']
    
    for mode in modes:
        iters = Bellman_Ford_variante(M, 0, mode)
        print("Mode", mode, ", itérations :", iters)


# -----------------------------------------------------------------------------
# 6. Comparaison expérimentale des complexités
# -----------------------------------------------------------------------------

# 6.1 Temps de calcul
def TempsDij(n):
    """
    Calcule le temps d'exécution de l'algorithme de Dijkstra sur un graphe aléatoire.

    Args:
        n (int): Nombre de sommets du graphe.

    Returns:
        float: Durée d'exécution en secondes.
    """
    m = graphe3(n, 0.2, 1, 10)
    start = time.time()
    Dijkstra(m, 0)
    return time.time() - start

def TempsBF(n):
    """
    Calcule le temps d'exécution de l'algorithme de Bellman-Ford (variante) avec parcours en largeur.

    Args:
        n (int): Nombre de sommets du graphe.

    Returns:
        float: Durée d'exécution en secondes.
    """
    m = graphe3(n, 0.2, 1, 10)
    start = time.time()
    Bellman_Ford_variante(m, 0, mode='pl')
    return time.time() - start

# Des exemples pour les tests
valeurs = list(range(5, 201, 10))
temps_dij = [TempsDij(n) for n in valeurs]
temps_bf = [TempsBF(n) for n in valeurs]


# 6.2 Courbes et régression
def plot_temps_calcul_classique(valeurs, temps_dij, temps_bf):
    """
    Affiche un graphique comparatif du temps de calcul des algorithmes de Dijkstra et de Bellman-Ford (en mode PL).

    Args:
        valeurs (list[int]): Liste des tailles de graphes testées (valeurs de n).
        temps_dij (list[float]): Temps d'exécution de l'algorithme de Dijkstra pour chaque valeur de n.
        temps_bf (list[float]): Temps d'exécution de l'algorithme de Bellman-Ford (mode 'pl') pour chaque valeur de n.

    Returns:
        None: Cette fonction ne retourne rien, elle affiche un graphique.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(valeurs, temps_dij, label="Dijkstra", marker='o')
    plt.plot(valeurs, temps_bf, label="Bellman-Ford (PL)", marker='s')
    plt.xlabel("Taille du graphe (n)")
    plt.ylabel("Temps (secondes)")
    plt.title("Temps de calcul des plus courts chemins")
    plt.legend()
    plt.grid(True)
    plt.show()

def estime_exposant(valeurs, temps_dij, temps_bf):

    log_n = np.log(valeurs)
    log_dij = np.log(np.array(temps_dij) + 1e-8)
    log_bf = np.log(np.array(temps_bf) + 1e-8)
    result_dij = linregress(log_n, log_dij)
    result_bf = linregress(log_n, log_bf)
    print("Exposant estimé (Dijkstra) a ≈", result_dij.slope:.2f)
    print("Exposant estimé (Bellman-Ford PL) a ≈ ",result_bf.slope:.2f)

# 6.2 bis : cas p = 1/n
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

def plot_temps_calcul_variable(valeurs):
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

# -----------------------------------------------------------------------------
# 8. Forte connexité pour un graphe avec p=50%
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 9. Détermination du seuil de forte connexité
# -----------------------------------------------------------------------------

def seuil(n):
    p = 0.0
    while p <= 1.0:
        taux = teststatfc(n, p)
        if taux >= 99:
            return round(p, 2)
        p += 0.01
    return None

# -----------------------------------------------------------------------------
# 10. Étude et identification de la fonction seuil
# -----------------------------------------------------------------------------

# 10.1 Graphique
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

# 10.2 Régression log-log
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