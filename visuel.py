import random
import sae0202

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
    a = 500
    b = 0
    for i in range(a):
        if fc(graphepourfc(n, p)):
            b += 1
    return (100 * b / a)

def lafonctionquiaffiche(p=0.5):
    print("forte connexité avec proba de 1 :", p)
    for n in range(1, 16):
        a = teststatfc(n, p)
        print("n = ",n,": ",a," %")

lafonctionquiaffiche