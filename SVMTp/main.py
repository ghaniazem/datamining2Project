import numpy as np
import pandas as pd
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
# Déclaration de fonctions

def convert_data(df) -> list:
    y = []  #déclaration d'un tableau vide
    for index, line in df.iterrows():  
        cont = [
            print(line['policy_id']),
            int(line['area_cluster']),
            int(line['segment']),
            int(line['model']),
            int(line['fuel_type']),
            int(line['max_torque']),
            int(line['max_power']),
            int(line['engine_type']),
            int(line['is_esc']),
            float(line['is_ajustable_steering']),
            int(line['is_tpms']),
            int(line['is_parking_sensors']),
            int(line['is_parking_camera']),
            int(line['rear_brakes_type']),
            int(line['transmission_type']),
            int(line['gear_bocks']),
            int(line['height']),
            int(line['gross_weigth']),
            int(line['is_front_for_lights']),
            int(line['is_rear_window_wiper']),
            int(line['is_rear_window_washer']) ,
            int(line['is_rear_window_defogger']),
            int(line['is_brake_assist']) ,
            int(line['is_power_door_loks']) ,
            int(line['is_central_locking']) ,
            int(line['is_power_steering']),
            int(line['is_driver_seat_height_adjustable']) ,
            int(line['is_day_night_rear_view_mirror'])

            ]
        y.append(cont)


    return y








if __name__ == '__main__':
    #importation de données
    df = pd.read_csv("sample_submission.csv")
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

    # afficher les dimensions de DataFrame (nombre de lignes et nombre de colonnes)
    print("dimension de DataFrame: {} rows, {} columns".format(df.shape[0], df.shape[1]))
    print("dimension de DataFrame de teste : {} rows, {} columns".format(df_test.shape[0], df_test.shape[1]))
    print("dimension de DataFrame de train : {} rows, {} columns".format(df_train.shape[0], df_train.shape[1]))

    # Générer des statistiques descriptives
    print(df.describe())
    print(df_test.describe())
    print(df_train.describe())

    # Renvoyer le nombre de valeurs uniques de chaque classe de "is_claim"
    print("Les valeurs de classe is_claim est \n ",df_train['is_claim'].value_counts(), '\n')

    # Renvoyer le pourcentage des voitures avec ou sans assurance dans  chaque classe
    count_no_is_claim = len(df_train[df_train.is_claim == 0])
    count_is_claim = len(df_train[df_train.is_claim == 1])
    print("Pourcentage des voitures sans assurance: {:.2f}%".format(
        (count_no_is_claim / (len(df_train.is_claim)) * 100)))
    print("Pourcentage des voitures avec assurance : {:.2f}%".format(
        (count_is_claim / (len(df_train.is_claim)) * 100)), "\n")

    #conversion de données
    data_train_convertis = convert_data(df_train)
    data_test_convertis = convert_data(df_test)
    print(data_train_convertis)
    #conversion en liste
    l = np.array(df)
    l_test = np.array(df_test)
    l_train = np.array(df_test)

    #print(l)
    #print(l_test)
    t = list(l_train[::1])
    #print("liste ",t)

