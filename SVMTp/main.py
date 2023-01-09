import numpy as np
import pandas as pd
from typing import Iterable

#asssss
class SVM:
    
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, nbr_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = nbr_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, x):
        n_features = x.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    
    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1


    def _get_gradients(self, constrain, x, idx):
            
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db


    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
        
        

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)



def convert_data(df: pd.DataFrame):

    # les données catégorique sont de type object
    categorical_features = [cols for cols in df if df[cols].dtype == 'object']
    #print(categorical_features)

    # transformer le données catégorie en numérique
    df_train_new1 = pd.get_dummies(data=df, columns=[
        'area_cluster',
        'segment',
        'model',
        'fuel_type',
        'max_torque',
        'max_power',
        'engine_type',
        'is_esc',
        'is_adjustable_steering',
        'is_tpms',
        'is_parking_sensors',
        'is_parking_camera',
        'rear_brakes_type',
        'transmission_type',
        'steering_type',
        'is_front_fog_lights',
        'is_rear_window_wiper',
        'is_rear_window_washer',
        'is_rear_window_defogger',
        'is_brake_assist',
        'is_power_door_locks',
        'is_central_locking',
        'is_power_steering',
        'is_driver_seat_height_adjustable',
        'is_day_night_rear_view_mirror',
        'is_ecw',
        'is_speed_alert'
    ], drop_first=True)

    #print(df_train_new1.head(3))

    # supprimer colonne d'identifiant pas important pour entrainement
    df_train_new2 = df_train_new1.drop(columns='policy_id', axis=1)

    return df_train_new2

def accuracy(predictions : Iterable, trues : Iterable):
    cpt = 0
    for p, t in zip(predictions, trues):
        if p == t:
            cpt += 1
    return cpt / len(predictions)





if __name__ == '__main__':
    #importation de données
    df = pd.read_csv("sample_submission.csv")
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

    #afficher les dimensions de DataFrame (nombre de lignes et nombre de colonnes)
    print("dimension de DataFrame: {} rows, {} columns".format(df.shape[0], df.shape[1]), '\n')
    print("dimension de DataFrame de teste : {} rows, {} columns".format(df_test.shape[0], df_test.shape[1]), '\n')
    print("dimension de DataFrame de train : {} rows, {} columns".format(df_train.shape[0], df_train.shape[1]), '\n')

    # Générer des statistiques descriptives
    print("statistiques descriptives de df", df.describe(), '\n')
    print("statistiques descriptives de df_test",df_test.describe(), '\n')
    print("statistiques descriptives de df_train",df_train.describe(), '\n \n')

    #afficher des informations sur les donees
    print("Informations sur le dataset : \n \n")
    print(df_train.info(), '\n \n')

    #Renvoyer le nombre de valeurs uniques de chaque classe de "is_claim"
    print("Les valeurs de classe is_claim est \n ",df_train['is_claim'].value_counts(), '\n')

    # Renvoyer le pourcentage des clients qui vont faire ou pas une demande de remboursement à son assurance automobile
    count_no_is_claim = len(df_train[df_train.is_claim == 0])
    count_is_claim = len(df_train[df_train.is_claim == 1])
    print("Pourcentage des clients qui vont faire une demande de remboursement: {:.2f}%".format(
       (count_no_is_claim / (len(df_train.is_claim)) * 100)))
    print("Pourcentage des clients qui vont pas faire une demande de remboursement : {:.2f}%".format(
       (count_is_claim / (len(df_train.is_claim)) * 100)), "\n")

    #conversion de données
    df_test_converted = convert_data(df_test)
    df_train_converted = convert_data(df_train)

    #conversion en liste

    l = np.array(df)
    l_test = np.array(df_test_converted)
    #print(l_test, '\n')
    l_train = np.array(df_train_converted)

    #mélnge de données
    np.random.shuffle(l_test)
    np.random.shuffle(l_train)

    #Normalisation de données (test)
    moy = np.mean(l_test, axis=0)
    dev = np.std(l_test, axis=0)
    l_test = (l_test - moy) / dev
    #print(l_test, '\n')

    # Normalisation de données (train)
    X_train = l_train[:, 0:43] #liste qui contient toutes les colonnes sauf la classe
    Y_train = l_train[:, 43] #liste qui contient que la classe
    #print(X_train)
    moy_train = np.mean(X_train, axis=0)
    dev_train = np.std(X_train, axis=0)
    X_train = (X_train - moy_train) / dev_train

    #création d'un modèle SVM

    model = SVM()
    print(model)

    #entrainement de modele
    #model.fit(X_train, Y_train)

    #predictions sur les données de test
    predictions = model.predict(l_test)

    #calcul d'accuracy
    acc = accuracy(predictions, Y_train)


