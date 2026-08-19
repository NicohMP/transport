# Optimisation du solveur de transport SN

## Objectif

Ce document décrit les optimisations appliquées au solveur stationnaire de
transport SN et au calcul de criticité dans `diffusion.ipynb`.

Les objectifs sont les suivants :

- réduire le nombre total de balayages de transport ;
- réduire les passages entre Python, NumPy et Numba ;
- limiter les allocations et copies de tableaux ;
- conserver la positivité du flux ;
- employer un critère d'arrêt fondé sur l'équation de criticité ;
- obtenir un solveur suffisamment rapide pour générer des données destinées
  à l'apprentissage automatique.

Le cas de référence utilisé pendant les mesures contient 1000 cellules et 16
directions de Gauss-Legendre. Le maillage à 10000 cellules a également été
mesuré pour vérifier le passage à l'échelle.

## 1. Équation résolue

Le problème de criticité en transport s'écrit

$$
\mu\,\partial_x\psi + \Sigma_t\psi
= \frac{\Sigma_s}{2}\phi
+ \frac{1}{k}\frac{\nu\Sigma_f}{2}\phi,
\qquad
\phi(x)=\int_{-1}^{1}\psi(x,\mu)\,\mathrm d\mu.
$$

Après discrétisation angulaire,

$$
\phi_i = \sum_{m=1}^{M} w_m\psi_{m,i}.
$$

Pour une direction positive, le schéma amont donne

$$
\psi_{m,i}
= \frac{\mu_m\psi_{m,i-1}+hS_i}
       {\mu_m+h\Sigma_{t,i}}.
$$

Pour une direction négative, le parcours spatial est inversé :

$$
\psi_{m,i}
= \frac{|\mu_m|\psi_{m,i+1}+hS_i}
       {|\mu_m|+h\Sigma_{t,i}}.
$$

La dépendance à la valeur amont impose un parcours séquentiel en espace pour
chaque direction. En revanche, les directions d'un même signe sont
indépendantes les unes des autres à position fixée.

## 2. Séparation des sources de diffusion et de fission

Pour une itération externe donnée, la source de fission est figée :

$$
Q_f^n = \frac{1}{k^n}\frac{\nu\Sigma_f}{2}\phi^n.
$$

Le problème interne est donc

$$
(T-S)\psi^{n+1}=Q_f^n.
$$

Dans le code :

```python
fixed_source = fission_source_coefficient * phi / k

psi_new, phi_new, inner_iter, inner_error = problem.solve_flux_only(
    psi0=psi,
    fixed_source=fixed_source,
    source_operator=mesh.Sigma_s,
    phi_tol=inner_tol,
    assume_valid=True,
)
```

Il est important de passer `Sigma_s` comme `source_operator`. Employer
`Sigma_p = Sigma_s + nu * Sigma_f` compterait à nouveau la fission dans le
problème interne et modifierait l'équation propre résolue.

Cette modification est une correction physique avant d'être une
optimisation.

## 3. Compilation de l'itération interne complète

Le noyau `solve_flux_only_numba` se trouve dans
`transport/sn/_kernels.py`. Le fichier `stationary.py` conserve l'API du
problème et délègue les boucles compilées à ce module privé.

Les invariants de la quadrature (tailles, finitude, ordre pair, signe et
symétrie des nœuds et des poids) sont validés à la construction de
`SNQuadrature`. Le problème stationnaire ne valide donc plus que les
conditions aux bords qui lui sont propres.

Avant l'optimisation, Python reprenait le contrôle à chaque étape :

```python
phi = scalar_flux(psi)
source = 0.5 * source_operator * phi + fixed_source
psi_new = sweep(source)
phi_new = scalar_flux(psi_new)
error = ...
```

Les sweeps étaient compilés, mais la boucle d'itération sur la source restait
en Python. Elle provoquait de nombreux appels, allocations et opérations
NumPy de petite taille.

Le noyau actuel compile en une seule fonction :

1. le calcul du flux scalaire initial ;
2. la construction de la source ;
3. le balayage négatif ;
4. le balayage positif ;
5. la quadrature donnant le nouveau flux scalaire ;
6. le calcul de l'erreur interne ;
7. le test de convergence.

La boucle complète s'exécute donc en code machine sans retour vers Python à
chaque sweep.

## 4. Ordre des directions négatives

Les nœuds de Gauss-Legendre sont stockés dans l'ordre

$$
-\mu_{\max},\ldots,-\mu_{\min},
\mu_{\min},\ldots,\mu_{\max}.
$$

Le tableau `mu_pos` contient

$$
\mu_{\min},\ldots,\mu_{\max}.
$$

Pour retrouver la valeur absolue correspondant à la ligne négative `m`, le
noyau utilise donc

```python
mu_abs = mu_pos[n_half - 1 - m]
```

Cette inversion est indispensable pour conserver l'ordre original des lignes
de `psi` après la réunion des directions négatives et positives.

## 5. Réutilisation des tableaux de travail

Les tableaux temporaires sont alloués une seule fois au début de
`_solve_flux_only_numba` :

```python
phi = np.empty(n_cells)
phi_new = np.empty(n_cells)
source = np.empty(n_cells)
psi_new = np.empty_like(psi)
incoming_positive = np.empty(n_half)
incoming_negative = np.empty(n_half)
```

À la fin d'une itération, les références sont échangées :

```python
psi, psi_new = psi_new, psi
phi, phi_new = phi_new, phi
```

Cette opération ne copie aucune donnée. Le tableau contenant l'ancien itéré
devient le tableau de travail de l'itération suivante.

Les gains viennent de la réduction :

- des allocations répétées ;
- des concaténations des demi-flux ;
- des tableaux temporaires produits par les opérations NumPy ;
- du trafic mémoire associé.

## 6. Fusion du calcul de phi et de l'erreur

Le nouveau flux scalaire est calculé cellule par cellule :

```python
for i in range(n_cells):
    value = 0.0
    for m in range(n_directions):
        value += weights[m] * psi_new[m, i]
    phi_new[i] = value
```

La même boucle met à jour les maxima nécessaires au critère interne :

$$
\varepsilon_{\mathrm{int}}
= \frac{\max_i|\phi_i^{\ell+1}-\phi_i^\ell|}
       {\max_i|\phi_i^\ell|}.
$$

Cela évite de former explicitement `phi_new - phi` et d'appeler plusieurs fois
`np.linalg.norm`.

## 7. Cache Numba

Les noyaux utilisent

```python
@njit(cache=True)
```

Numba peut ainsi conserver la compilation sur disque et la recharger dans un
nouveau processus lorsque les signatures sont identiques.

Il faut distinguer deux temps :

- le temps du premier appel, qui peut inclure la compilation ou le chargement
  du cache ;
- le temps après préchauffage, qui représente le débit utile lors de la
  génération d'un grand nombre de cas.

Pour profiler le débit réel, il faut toujours effectuer un appel de
préchauffage avant d'activer le chronomètre ou `cProfile`.

## 8. Chemin rapide assume_valid

Par défaut, `solve_flux_only` :

- convertit les entrées en tableaux flottants ;
- copie `psi0` ;
- contrôle les dimensions ;
- recherche les valeurs non finies.

Ces vérifications sont adaptées à une API publique, mais deviennent inutiles
quand la même boucle de criticité appelle la méthode des centaines de fois
avec des tableaux déjà contrôlés.

L'option

```python
assume_valid=True
```

supprime alors les copies et validations répétées.

Attention : dans ce mode, le noyau peut réutiliser `psi0` comme tableau de
travail. L'appelant doit remplacer immédiatement l'ancien flux par la valeur
retournée. Ce mode ne doit pas être utilisé avec un tableau en lecture seule
ou avec un tableau que l'on souhaite conserver intact.

Le comportement sécurisé reste le comportement par défaut.

## 9. Tolérance interne adaptative

Résoudre très précisément les premières équations internes est inutile : la
source de fission externe est encore loin du mode fondamental.

La tolérance est donc adaptée à l'erreur externe courante :

```python
inner_tol = max(
    inner_tol_floor,
    min(1e-2, 0.1 * outer_error),
)
```

Au début, la tolérance est proche de `1e-2`. Elle diminue progressivement
jusqu'à la précision finale demandée.

Il s'agit d'une résolution inexacte emboîtée :

$$
\text{précision du problème interne}
\approx
\text{précision externe courante}.
$$

Sur le cas de référence, cette seule stratégie a fait passer le nombre total
de sweeps d'environ 14084 à environ 5100.

## 10. Initialisation par le problème de diffusion

Dans un milieu diffusant, le mode propre de diffusion fournit une bonne
approximation du mode propre de transport.

Le notebook calcule donc d'abord :

```python
diffusion_guess = criticity_factor(
    mesh,
    ktol=1e-8,
    residual_tol=1e-6,
)
phi = diffusion_guess.phi.copy()
k = diffusion_guess.k_eff
```

Le flux angulaire initial est pris isotrope :

```python
psi = np.broadcast_to(
    0.5 * phi,
    (quadrature.order, mesh.n_cells),
).copy()
```

Le solveur de transport commence ainsi près du mode fondamental au lieu de
partir d'un flux spatialement constant.

Cette initialisation est contrôlée par

```python
use_diffusion_guess=True
```

et peut être désactivée pour étudier la convergence brute de l'itération de
puissance.

## 11. Sur-relaxation externe

Après une application de l'opérateur de puissance, on obtient un itéré brut

$$
(\phi_{\mathrm{raw}}^{n+1},\psi_{\mathrm{raw}}^{n+1},k_{\mathrm{raw}}^{n+1}).
$$

Une extrapolation est appliquée :

$$
\phi^{n+1}
= \phi^n + \omega
  \left(\phi_{\mathrm{raw}}^{n+1}-\phi^n\right),
$$

et de manière analogue pour `psi` et `k`. Le coefficient utilisé par défaut
est

$$
\omega=1.5.
$$

Dans le code :

```python
phi_relaxed = phi + relaxation * (phi_new - phi)
psi_relaxed = psi + relaxation * (psi_new - psi)
k_relaxed = k + relaxation * (k_raw - k)
```

Une extrapolation peut créer des valeurs négatives. Elle n'est donc acceptée
que si

```python
k_relaxed > 0.0
np.min(phi_relaxed) >= 0.0
np.min(psi_relaxed) >= 0.0
```

Sinon, l'itéré non relaxé est conservé. La première itération utilise toujours
`omega = 1` afin de construire une forme angulaire compatible avec les
conditions au bord.

Le coefficient est limité à l'intervalle `[1, 2)`. La valeur `1` désactive la
sur-relaxation.

Sur le cas de référence, cette méthode a réduit le nombre d'itérations
externes d'environ 1070 à environ 810.

## 12. Résidu normalisé de criticité

La différence entre deux flux successifs n'est pas suffisante pour vérifier
la résolution de l'équation propre. Le résidu utilisé est

$$
r_k = (T-S)\psi - \frac{1}{k}F\psi.
$$

Sa version normalisée est

$$
\varepsilon_r
= \frac{\|r_k\|_2}
       {\|(T-S)\psi\|_2 + \|F\psi/k\|_2}.
$$

La méthode `StatTransportProblem.criticality_residual` calcule ce résidu sans
assembler de matrice de taille `M*N`. Elle reconstruit directement le terme de
transport amont dans chaque cellule et chaque direction.

Pour une direction positive, le terme discret est

$$
(\mu_m+h\Sigma_{t,i})\psi_{m,i}
-\mu_m\psi_{m,i-1}.
$$

Pour une direction négative, la valeur amont vient de la cellule située à
droite.

L'arrêt externe impose désormais

```python
err_k < ktol and normalized_residual < residual_tol
```

La variation de `phi` n'est plus utilisée comme preuve finale de convergence.

## 13. Évaluation intermittente du résidu

Calculer le résidu exact demande une traversée complète du tableau `psi`. Son
coût est comparable à celui d'un sweep.

Il est donc évalué toutes les dix itérations par défaut :

```python
residual_checked = (
    outer_iteration == 1
    or outer_iteration % residual_check_interval == 0
)
```

Entre deux évaluations, la dernière valeur connue sert à régler la tolérance
interne. L'algorithme ne peut s'arrêter que sur une itération où le résidu a
réellement été recalculé.

Cette stratégie ne modifie pas le critère mathématique. Elle peut seulement
retarder sa détection de `residual_check_interval - 1` itérations au maximum.

La courbe du résidu dans le notebook présente donc naturellement des paliers
entre deux évaluations.

## 14. Pré-calcul des quantités invariantes

Les coefficients indépendants des itérations sont calculés une seule fois :

```python
fission_weight = mesh.h * mesh.nu * mesh.Sigma_f
fission_source_coefficient = 0.5 * mesh.nu * mesh.Sigma_f
```

La production de fission est alors

$$
\mathcal P(\phi)
= (h\nu\Sigma_f)^T\phi.
$$

Dans le code :

```python
prod_new = fission_weight @ phi_new
```

Le flux est ensuite normalisé :

```python
phi_new /= prod_new
psi_new /= prod_new
```

Par construction,

$$
\mathcal P(\phi^n)=1
$$

à chaque itération. Le dénominateur de l'ancienne formule de mise à jour vaut
donc toujours 1 et

```python
k_new = k * prod_new / prod_old
```

se simplifie en

```python
k_raw = k * prod_new
```

## 15. Chemin de calcul final

Une itération externe suit maintenant la chaîne suivante :

```text
source de fission
    -> tolérance interne adaptative
    -> résolution interne entièrement compilée
    -> production de fission
    -> normalisation de phi et psi
    -> sur-relaxation avec contrôle de positivité
    -> calcul périodique du résidu propre
    -> test de convergence
```

## 16. Résultats mesurés

Les temps suivants sont des temps après préchauffage de Numba. Ils dépendent
de la machine, mais les nombres d'itérations sont reproductibles pour les
paramètres du notebook.

| Maillage | Directions | Temps total | Itérations externes | Itérations internes cumulées | Résidu final |
|---:|---:|---:|---:|---:|---:|
| 1000 | 16 | environ 0.074 s | 810 | 1712 | 9.96e-7 |
| 10000 | 16 | environ 0.36 s | 650 | 902 | 9.84e-7 |

Pour `N=1000`, le temps observé avant ces optimisations était compris entre
environ 0.35 s et 0.41 s selon la variante profilée. Le gain global est donc
de l'ordre de cinq sur le cas de référence, tout en utilisant maintenant un
critère d'arrêt plus rigoureux.

Le nombre d'itérations internes peut être inférieur sur le maillage fin parce
que l'initialisation de diffusion y représente mieux le mode spatial et que de
nombreux problèmes internes convergent en un seul sweep.

## 17. Lecture correcte d'un profil cProfile avec Numba

`cProfile` ne détaille pas les boucles exécutées dans le code machine Numba.
Le temps du noyau apparaît donc généralement sur la ligne de la méthode Python
qui l'appelle :

```text
StatTransportProblem.solve_flux_only
```

Cela ne signifie pas que le corps Python de la méthode est lent. Cette ligne
inclut le temps passé dans `_solve_flux_only_numba`.

Procédure recommandée :

```python
# Préchauffage Numba
problem.solve_flux_only(...)
problem.criticality_residual(...)

profiler = cProfile.Profile()
profiler.enable()
result = transport_criticity(mesh, quadrature)
profiler.disable()
```

Pour mesurer précisément un noyau individuel, utiliser également
`time.perf_counter()` et répéter plusieurs fois la mesure. La médiane est plus
représentative que la première exécution.

## 18. Vérifications numériques

Le fichier `tests/test_sn_stationary.py` vérifie :

- que `solve_flux_only` reproduit `solve` pour `psi`, `phi`, l'erreur et le
  nombre d'itérations ;
- que le chemin `assume_valid=True` donne le même résultat ;
- que le résidu compilé correspond à une construction indépendante de
  l'opérateur discret.
- que `TransportStationaryResult.validate(...)` contrôle les invariants du
  résultat et la tolérance finale.

Exécution :

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Les écarts entre le chemin complet et le chemin optimisé sont inférieurs à
`1e-14` dans les tests actuels.

## 19. Optimisations testées mais non retenues

### Former deux sweeps puis les concaténer

Cette version est simple et reste utilisée par la méthode générale `sweep`.
Dans la boucle interne de criticité, les allocations des deux demi-flux et la
concaténation répétée sont toutefois moins efficaces qu'un noyau unique.

### Pré-calculer tous les dénominateurs du sweep

Les coefficients

$$
\frac{\mu_m}{\mu_m+h\Sigma_t}
\quad\text{et}\quad
\frac{h}{\mu_m+h\Sigma_t}
$$

peuvent être pré-calculés. Sur le cas testé, les lectures mémoire
supplémentaires annulaient presque le gain lié à la suppression des divisions.
Cette option n'a donc pas été retenue.

### Paralléliser les cellules spatiales

Cette transformation n'est pas directement possible avec le schéma amont,
car chaque cellule dépend de la sortie de la cellule précédente. Une méthode
de type scan parallèle changerait fortement la complexité et n'est pas
justifiée pour les tailles actuelles.

### Déplacement de toute la criticité dans un noyau Numba

Cette approche réduirait encore les appels Python externes, mais mélangerait
le solveur stationnaire, l'itération propre, l'initialisation de diffusion et
les diagnostics dans un seul noyau difficile à maintenir.

Le profil final attribue déjà la majorité du temps au calcul physique. Le gain
maximal attendu de cette fusion est nettement plus faible que ceux obtenus par
la tolérance adaptative, l'initialisation de diffusion et la compilation de
l'itération interne.

### Décalage de Wielandt

Le décalage réduit le nombre d'itérations externes, mais rapproche l'opérateur
interne de la singularité et augmente fortement le nombre d'itérations de
diffusion. Sans accélération interne de type DSA, le temps total n'était pas
meilleur sur le cas étudié.

## 20. Prochaines optimisations pertinentes

### Warm-start entre problèmes voisins

Pour créer un dataset ML, les paramètres peuvent être ordonnés de façon à ce
que deux réacteurs successifs soient proches. Le triplet convergé

$$
(k,\phi,\psi)
$$

du problème précédent peut servir d'initialisation au suivant. Cette stratégie
peut être plus efficace que toute micro-optimisation supplémentaire.

### Diffusion Synthetic Acceleration

Lorsque `Sigma_s / Sigma_t` est très proche de 1, l'itération sur la source
reste lente. Une DSA corrige les modes spatiaux lentement amortis à l'aide d'un
problème de diffusion auxiliaire.

Cette méthode devient prioritaire si les profils futurs montrent à nouveau un
grand nombre d'itérations internes par résolution.

### Continuation de maillage

Une solution peut être calculée sur un maillage grossier, interpolée, puis
utilisée comme initialisation sur un maillage fin. Cette approche est
particulièrement intéressante pour les calculs à haute résolution destinés à
produire des références ML.

## 21. Paramètres de contrôle importants

La fonction du notebook expose notamment :

```python
transport_criticity(
    mesh,
    quadrature,
    ktol=1e-6,
    residual_tol=1e-6,
    use_diffusion_guess=True,
    outer_relaxation=1.5,
    residual_check_interval=10,
    max_outer_iterations=10_000,
)
```

- `ktol` contrôle la variation relative de `k` ;
- `residual_tol` contrôle la satisfaction de l'équation propre ;
- `use_diffusion_guess` active l'initialisation par diffusion ;
- `outer_relaxation=1` désactive la sur-relaxation ;
- `residual_check_interval=1` recalcule le résidu à chaque itération ;
- `max_outer_iterations` protège contre une boucle infinie.

Pour produire des références de haute précision, il faut diminuer à la fois
`ktol` et `residual_tol`. Le temps de calcul augmente alors parce que le mode
propre doit converger davantage et que les problèmes internes sont résolus
plus précisément.
