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
        
        

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)



def convert_data():
    df = pd.read_csv("train.csv")

    # les données catégorique sont de type object
    categorical_features = [cols for cols in df if df[cols].dtype == 'object']
    print(categorical_features)

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

    print(df_train_new1.head(3))

    # supprimer colonne d'identifiant pas important pour entrainement
    df_train_new2 = df_train_new1.drop(columns='policy_id', axis=1)
    print(df_train_new2.head(3))

    df = pd.read_csv("test.csv")
    # print(df['policy_id'])

    # les données catégorique sont de type object
    categorical_features = [cols for cols in df if df[cols].dtype == 'object']

    # transformer le données catégorie en numérique
    df_test_new1 = pd.get_dummies(data=df, columns=[
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

    print(df_test_new1.head())

    # supprimer colonne d'identifiant pas important pour test
    df_test_new2 = df_test_new1.drop(columns='policy_id', axis=1)
    print(df_test_new2.head())





if __name__ == '__main__':
    #importation de données
    df = pd.read_csv("sample_submission.csv")
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

    #afficher les dimensions de DataFrame (nombre de lignes et nombre de colonnes)
    print("dimension de DataFrame: {} rows, {} columns".format(df.shape[0], df.shape[1]))
    print("dimension de DataFrame de teste : {} rows, {} columns".format(df_test.shape[0], df_test.shape[1]))
    print("dimension de DataFrame de train : {} rows, {} columns".format(df_train.shape[0], df_train.shape[1]))

    # Générer des statistiques descriptives
    print(df.describe())
    print(df_test.describe())
    print(df_train.describe())

    #Renvoyer le nombre de valeurs uniques de chaque classe de "is_claim"
    print("Les valeurs de classe is_claim est \n ",df_train['is_claim'].value_counts(), '\n')

    # Renvoyer le pourcentage des voitures avec ou sans assurance dans  chaque classe
    count_no_is_claim = len(df_train[df_train.is_claim == 0])
    count_is_claim = len(df_train[df_train.is_claim == 1])
    print("Pourcentage des voitures sans assurance: {:.2f}%".format(
       (count_no_is_claim / (len(df_train.is_claim)) * 100)))
    print("Pourcentage des voitures avec assurance : {:.2f}%".format(
       (count_is_claim / (len(df_train.is_claim)) * 100)), "\n")

    #conversion de données
    #data_train_convertis = convert_data(df_train)
    #print(data_train_convertis)

    #conversion en liste
    l = np.array(df)
    l_test = np.array(df_test)
    l_train = np.array(df_test)

    #print(l)
    #print(l_test)
    t = list(l_train[::1])
    #print("liste ",t)

