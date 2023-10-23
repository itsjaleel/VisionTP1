## Pipeline complète de classification d’images

1-d-11) La distribution du bruit gaussien ressemble à une distribution en forme de cloche, ce qui est caractéristique d'une distribution gaussienne (normale). La forme de cette distribution fait penser à une cloche, d'où le nom de distribution gaussienne ou normale.



2-b) Il y a 916 images dans le dossier.

2-c) Pour une image choisi au hasard: Le format de l image est .jpeg et la taille de l image est 4272 octets.

2-f-3) L'argument `random_state` sert à initialiser la génération de nombres aléatoires. Il permet de reproduire les mêmes résultats dans une opération aléatoire, ce qui est utile pour la reproductibilité des expériences.

2-a-4) Pour prédire le label de la première image dans l'ensemble de test, nous avons utilisé la méthode predict. La prédiction a été affichée à l'écran.

3-c-2) Matrice de Confusion pour modèle 1:
[[11  2]
[ 0  8]]

En haut à gauche (11), c'est le nombre de vrais négatifs (TN). Cela signifie que le modèle a correctement prédit 11 cas négatifs.
En haut à droite (2), c'est le nombre de faux positifs (FP). Cela signifie que le modèle a prédit à tort 2 cas positifs qui étaient en réalité négatifs.
En bas à gauche (0), c'est le nombre de faux négatifs (FN). Cela signifie que le modèle a prédit à tort 0 cas négatifs qui étaient en réalité positifs.
En bas à droite (8), c'est le nombre de vrais positifs (TP). Cela signifie que le modèle a correctement prédit 8 cas positifs.
