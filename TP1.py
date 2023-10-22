# %%
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img1
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix





def peuplate_images_and_labels_lists(image_folder_path):
    images= []
    labels = []
    label = os.path.basename(os.path.normpath(image_folder_path))
    for filename in os.listdir(image_folder_path):
        image = cv2.imread(os.path.join(image_folder_path, filename))
        if image is not None:
            image = cv2.resize(image, target_size)
            images.append(image)
            labels.append(label)
    return images, labels

def peuplate_and_augment_images_and_labels_lists(image_folder_path):
    images = []
    labels = []
    label = os.path.basename(os.path.normpath(image_folder_path))
    for filename in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Redimensionner l'image
            image = cv2.resize(image, target_size)
            
            # Transformation de cropping
            cropped_image = image[48:162, 48:162]
            images.append(cropped_image)
            labels.append(label)

            # Transformation en noir et blanc
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grey_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
            images.append(grey_image)
            labels.append(label)

    return images, labels


def aplatir_image(image):
    return image.flatten()

X = np.random.rand(1000) * 3
print(X[:10])

moyenne = np.mean(X)
ecart_type = np.std(X)
median = np.median(X)

moyenne_arrondie = round(moyenne, 2)
ecart_type_arrondi = round(ecart_type, 2)
median_arrondie = round(median, 2)

print(f"Moyenne : {moyenne_arrondie}")
print(f"Écart type : {ecart_type_arrondi}")
print(f"Médiane : {median_arrondie}")

X_bis = np.random.rand(1000) * 3

moyenne_bis = np.mean(X_bis)
ecart_type_bis = np.std(X_bis)
median_bis = np.median(X_bis)

moyenne_arrondie_bis = round(moyenne, 2)
ecart_type_arrondi_bis = round(ecart_type, 2)
median_arrondie_bis = round(median, 2)

print(f"Moyenne bis : {moyenne_arrondie_bis}")
print(f"Écart type bis: {ecart_type_arrondi_bis}")
print(f"Médiane bis: {median_arrondie_bis}")

np.random.seed(0)

X = np.random.rand(1000) * 3
print(X[:10])

moyenne = np.mean(X)
ecart_type = np.std(X)
median = np.median(X)

moyenne_arrondie = round(moyenne, 2)
ecart_type_arrondi = round(ecart_type, 2)
median_arrondie = round(median, 2)

print(f"Moyenne : {moyenne_arrondie}")
print(f"Écart type : {ecart_type_arrondi}")
print(f"Médiane : {median_arrondie}")

X_bis = np.random.rand(1000) * 3

moyenne_bis = np.mean(X_bis)
ecart_type_bis = np.std(X_bis)
median_bis = np.median(X_bis)

moyenne_arrondie_bis = round(moyenne, 2)
ecart_type_arrondi_bis = round(ecart_type, 2)
median_arrondie_bis = round(median, 2)

print(f"Moyenne bis : {moyenne_arrondie_bis}")
print(f"Écart type bis: {ecart_type_arrondi_bis}")
print(f"Médiane bis: {median_arrondie_bis}")

y = np.sin(X) + 0.1 * np.random.randn(1000)

print(y[:10])
#%%
#plt.figure(figsize=(8,6))
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=10, c='blue', alpha=0.5)
plt.scatter(X, y)
plt.show()

noise = np.random.randn(1000)

plt.hist(noise, bins=50, color='green', alpha=0.7)


plt.xlabel('Valeurs du Bruit Gaussien')
plt.ylabel('Fréquence')
plt.title('Histogramme du Bruit Gaussien')


plt.show()
# %%
chemin_dossier_bike = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/data1/bike'
chemin_dossier_car = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/data1/car'
images = []

for fichier in os.listdir(chemin_dossier_bike):
    chemin_fichier = os.path.join(chemin_dossier_bike, fichier)
    
    # Vérifier si le fichier est une image
    try:
        img = cv2.imread(chemin_fichier)
        if img is not None:
            images.append(fichier)
    except Exception as e:
        pass


for fichier in os.listdir(chemin_dossier_car):
    chemin_fichier = os.path.join(chemin_dossier_car, fichier)
    
    # Vérifier si le fichier est une image
    try:
        img = cv2.imread(chemin_fichier)
        if img is not None:
            images.append(fichier)
    except Exception as e:
        pass

nombre_images = len(images)

print(f'Il y a {nombre_images} images dans le dossier.')

# %%

image_bike = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/data1/bike/Bike (2).jpeg'
format_image = os.path.splitext(image_bike)[1].lower()
taille_image = os.path.getsize(image_bike)
print(f'Le format de l image est {format_image}.')
print(f'La taille de l image est {taille_image} octets.')
# %%

# %%
image = img1.imread(image_bike)
plt.axis('off')  # Pour supprimer les axes
plt.imshow(image)
plt.show()
# %%

plt.imshow(image[:,:,1], cmap="gray")
plt.show()
# %%

plt.imshow(image[::-1], origin='upper')
plt.axis('off')  # Pour supprimer les axes
plt.show()

# %%

bike_folder = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/data1/bike'
car_folder = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/data1/car'
target_size = (224,224)

# iv. Créer des array numpy pour les images et les labels
images_bike, labels_bike = peuplate_images_and_labels_lists(bike_folder)
images_car, labels_car = peuplate_images_and_labels_lists(car_folder)

# %%

# Convertir les listes en tableaux NumPy

labels = np.array(labels_bike + labels_car)
images = images_bike + images_car
print(images)
print('--------')
images = np.array([image.flatten() for image in images])
print(images)



# %%
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

# %%

# Définir l'arbre de décision
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Faire la prédiction
prediction_1 = clf.predict(X_test)

# Afficher la prédiction
print(f"La prédiction pour la première image est : {prediction_1}")

# %%
svc = SVC(random_state=0)
svc.fit(X_train, y_train)
prediction_2 = svc.predict(X_test)

# Afficher la prédiction
print(f"La prédiction pour la première image est : {prediction_2}")

# %%

# Calculer l'accuracy
accuracy = accuracy_score(y_test, prediction_1)

# Afficher l'accuracy
print(f"L'accuracy du modèle 1 est : {accuracy}")

# Calculer l'accuracy
accuracy = accuracy_score(y_test, prediction_2)
print(f"L'accuracy du modèle 2 est : {accuracy}")

# %% 
# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, prediction_1)

# Afficher la matrice de confusion
print("Matrice de Confusion pour modèle 1:")
print(conf_matrix)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, prediction_2)

# Afficher la matrice de confusion
print("Matrice de Confusion pour modèle 2:")
print(conf_matrix)


# %%
profondeur_arbre = clf.get_depth()
print(f"La profondeur de l'arbre de décision est : {profondeur_arbre}")

max_depth_list = list(range(1, 9))

train_accuracy = []
test_accuracy = []

for max_depth in max_depth_list:
    # Créer et entraîner un arbre de décision avec la profondeur donnée
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X_train, y_train)
    
    # Calculer l'accuracy sur les ensembles d'entraînement et de test
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Ajouter les accuracies aux listes correspondantes
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)



# Tracer les courbes
plt.plot(max_depth_list, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_list, test_accuracy, label='Test Accuracy')

# Ajouter des titres et labels d'axes
plt.title("Accuracy en fonction de max_depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")

# Ajouter une légende
plt.legend()

# Afficher le graphique
plt.show()

# %%

val_bike_folder = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/val/bike'
val_car_folder = '/Users/jaleelchoudry/Documents/esiee-it/Computer Vision/val/car'

# Créer les listes d'images et d'étiquettes de validation
val_images_bike, val_labels_bike = peuplate_images_and_labels_lists(val_bike_folder)
val_images_car, val_labels_car = peuplate_images_and_labels_lists(val_car_folder)



# Fusionner les images et les étiquettes pour l'ensemble de validation
val_images = np.concatenate((val_images_bike, val_images_car))
val_labels = np.concatenate((val_labels_bike, val_labels_car))

# Convertir les listes en tableaux NumPy

labels = np.array(val_labels_bike + val_labels_car)
images = val_images_bike + val_images_car
print(images)
print('--------')
images = np.array([image.flatten() for image in images])
print(images)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

# Définir l'arbre de décision
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Faire la prédiction
prediction_1 = clf.predict(X_test)

# Afficher la prédiction
print(f"La prédiction pour la première image est : {prediction_1}")

# Calculer l'accuracy
accuracy = accuracy_score(y_test, prediction_1)

# Afficher l'accuracy
print(f"L'accuracy du modèle 1 est : {accuracy}")
# %%
