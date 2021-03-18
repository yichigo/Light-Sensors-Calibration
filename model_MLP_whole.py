import pandas as pd
import numpy as np
import os
import sys
import datetime
import time

#from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler,Normalizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, r2_score

import matplotlib.dates as mdates
import matplotlib.colors
import matplotlib.ticker as ticker

from pysolar.solar import *
import pytz
import shap
import pickle
import multiprocessing

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

RANDOM_STATE = 42
n_jobs = multiprocessing.cpu_count()


cheap_node_list = ['001e063059c2', '001e06305a61', '001e06305a6c', '001e06318cd1',
                   '001e06323a05', '001e06305a57', '001e06305a6b', '001e06318c28',
                   '001e063239e3', '001e06323a12']

for cheap_node_id in cheap_node_list:
    print(cheap_node_id)

    # Perameters
    node_id = '10004098'
    gps_node_id = '001e0610c2e9'
    dir_out = '../figures/' + cheap_node_id + '/'
    dir_data = '../data/'

    years = ['2019','2020'] ####
    months = ['1','2','3','4','5','6','7','8','9','10','11','12']
    days = np.array(range(1,31+1)).astype(str) #### np.array(range(1,31+1)).astype(str)
    days = list(days)

    hours = (np.array(range(0,24))).astype(str)
    hours = list(hours)

    bins = np.array(range(0,420+1)).astype(str)
    bins = list(bins)
    for i in range(len(bins)):
        bins[i] = 'Spectrum[' + bins[i] + ']'

    wavelengths = np.array(range(360,780+1))#.astype(str)
    #for i in range(len(wavelengths)):
    #    wavelengths[i] = wavelengths[i] + 'nm'
    #wavelengths = list(wavelengths)


    # Read Data
    # if data has been preprocessed before, run this directly
    fn_data = dir_data + node_id + '_'+ cheap_node_id +'.csv'
    df = pd.read_csv(fn_data, parse_dates=True, index_col = 'UTC')
    df = df[(df.index.date != datetime.date(2019, 12, 31)) # Minolta was covered in these dates
           &(df.index.date != datetime.date(2019, 12, 27))
           &(df.index.date != datetime.date(2020,  1,  1))
           &(df.index.date != datetime.date(2020,  1,  2))]
    #        &(df.index.date != datetime.date(2020, 2, 14))
    #        &(df.index.date != datetime.date(2020, 2, 21))]


    features = [#'cloudPecentage', 'allRed', 'allGreen', 'allBlue',
            #'skyRed', 'skyGreen', 'skyBlue', 'cloudRed', 'cloudGreen', 'cloudBlue',
            'Violet', 'Blue', 'Green', 'Yellow', 'Orange', 'Red',
            'Temperature', 'Pressure', 'Humidity',
            #'Latitude', 'Longitude', 'Altitude',
            #'NH3', 'CO', 'NO2', 'C3H8', 'C4H10', 'CH4', 'H2', 'C2H5OH', 'CO2',
            'Luminosity', 'IR', 'Full', 'Visible', 'Lux',
            'UVA', 'UVB', 'Visible Compensation','IR Compensation', 'UV Index',
            'Zenith']
    features = np.array(features)
    len(features)

    targets = df.columns[-421-1:-1].values # skip Illuminance, keep Wavelengths
    print(features)
    print(targets[[0,-1]])

    X = df[features]
    Y = df[targets] # MLP and scaler use multi output


    ## Prepare Data  For Training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # scale the data
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_test_scaled = scaler_y.transform(Y_test)

    n_components = 18

    DR = 'PCA'
    Cluster = 'None'
    pca = PCA(n_components=n_components, random_state = RANDOM_STATE)

    X_train_scaled_DR = pca.fit_transform(X_train_scaled)
    X_test_scaled_DR = pca.transform(X_test_scaled)

    X_train_scaled_DR = pd.DataFrame(X_train_scaled_DR)
    X_test_scaled_DR = pd.DataFrame(X_test_scaled_DR)
    print(pca.explained_variance_)  #######

    # save pca model
    dir_DR = '../models/' + cheap_node_id + '/'
    if not os.path.exists(dir_DR):
        os.mkdir(dir_DR)
    fn_DR = dir_DR + DR + '.sav'
    pickle.dump(pca, open(fn_DR, 'wb'))

    # scale again
    scaler_x2 = StandardScaler()
    X_train_scaled_DR_scaled = scaler_x2.fit_transform(X_train_scaled_DR)
    X_test_scaled_DR_scaled = scaler_x2.transform(X_test_scaled_DR)

    ## Model for Whole Spectrum

    hidden_layer_sizes=(64,128,256)
    #hidden_layer_sizes=(128,128,128,128)
    #hidden_layer_sizes=(512, 512, 256, 256)
    #hidden_layer_sizes=(128,128,128,128,128,128)
    #hidden_layer_sizes=(128,128,128,128,128)

    activation ='relu'
    solver = 'adam'
    alpha=1e-5 # L2 penalty (regularization term) parameter, default 1e-5
    learning_rate = 'constant'

    # include layer structure and activation function
    structure = '_' + DR + str(n_components) + \
                '_' + str(hidden_layer_sizes)[1:-1].replace(', ','_') + \
                '_' + activation



    start_time = time.time()

    regr = MLPRegressor(random_state = RANDOM_STATE,
                        hidden_layer_sizes = hidden_layer_sizes,
                        activation = activation,
                        solver = solver,
                        alpha = alpha,
                        learning_rate = learning_rate,
                        verbose = True
                        )
    regr.fit(X_train_scaled_DR_scaled, Y_train_scaled)

    # fine tune the model
    regr.warm_start = True
    regr.learning_rate_init /= 10 # default 0.001
    regr.fit(X_train_scaled_DR_scaled, Y_train_scaled)

    regr.learning_rate_init /= 10 # default 0.001
    regr.fit(X_train_scaled_DR_scaled, Y_train_scaled)
    print("--- %s seconds ---" % (time.time() - start_time))


    # save model
    dir_model = '../models/' + cheap_node_id + '/'
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    dir_model += 'whole/'
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)

    fn_model = dir_model + 'MLP_whole' + structure + '.sav'
    pickle.dump(regr, open(fn_model, 'wb'))



    from sklearn.metrics import r2_score

    Y_train_pred = scaler_y.inverse_transform(
                        regr.predict( X_train_scaled_DR_scaled )
                        ) # for train
    Y_test_pred = scaler_y.inverse_transform(
                        regr.predict( X_test_scaled_DR_scaled )
                        )# for image
    #Y_test_pred = regr.predict(X_test) # for test score



    # Plot performance
    train_score =  r2_score(Y_train, Y_train_pred)
    test_score = r2_score(Y_test, Y_test_pred)

    y_min = np.amin(Y_train.values)
    y_max = np.amax(Y_train.values)
    y_line = np.linspace(y_min,y_max,100)

    plt.rcParams["figure.figsize"] = (8, 8) # (w, h)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    plt.plot(y_line,y_line, '-k', label='y=x')

    length_sample = len(Y_test)//10
    plt.scatter(Y_train[:length_sample],Y_train_pred[:length_sample], s=1, c = 'blue',label = 'Train, R$^{2}$ ='+str(train_score)[:6])
    plt.scatter(Y_test[:length_sample],Y_test_pred[:length_sample], s=1, c = 'red', label = 'Test, R$^{2}$ ='+str(test_score)[:6])
    plt.xlim((y_min,y_max))
    plt.ylim((y_min,y_max))
    ax.set_title('Predicted vs Actual for Whole Spectrum')
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    plt.legend( loc='lower right')
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(dir_out + 'whole'):
        os.mkdir(dir_out + 'whole')
    plt.savefig(dir_out + 'whole' +'/MLP_performance_whole'+structure+'.png')
    plt.close()


    # training score distribution
    train_scores = []
    for i in range(len(wavelengths)):
        train_scores.append( r2_score(Y_train.iloc[:,i], Y_train_pred[:,i]) )

    plt.rcParams["figure.figsize"] = (10, 5) # (w, h)
    plt.plot(np.array(range(360,780+1)),train_scores, 'k')
    plt.ylim(0.995,1)
    plt.title("R$^2$ Distribution")
    plt.savefig(dir_out + 'whole' + '/MLP_R2' + structure + '.png')
    plt.close()



    # Create object that can calculate shap values
    num_shap = 50
    explainer = shap.KernelExplainer(regr.predict, X_train_scaled_DR_scaled[:num_shap])
    # Calculate shap_values
    shap_values = explainer.shap_values(X_train_scaled_DR_scaled[:num_shap])


    # summary_plot
    max_display = n_components
    shap.summary_plot(np.mean(shap_values, axis = 0), X_train_scaled_DR_scaled[:num_shap],
                      plot_size=(10,max_display/2.5),#'auto'
                      max_display = max_display,
                      show=False,
                      plot_type = 'dot'
                     )
    plt.tight_layout()
    plt.savefig(dir_out + 'whole' + '/MLP_shap_whole' + structure + '.png')
    plt.close()


    # PCA Feature Importances
    shap.summary_plot(np.mean(shap_values, axis = 0), X_train_scaled_DR_scaled[:num_shap],
                  plot_size=(10,max_display/2.5),#'auto'
                  color = 'blue',
                  max_display = max_display,
                  show=False,
                  plot_type = 'bar'
                 )
    plt.xscale("log")
    plt.title('PCA Feature Importances for Whole Spectrum')
    plt.tight_layout()
    plt.savefig(dir_out + 'whole' + '/MLP_PCAImportances_whole' + structure +'.png')
    plt.close()


    # Feature Importances
    # rank feature importance
    num_features = len(features)
    importances_pca = np.mean(np.abs(np.mean(shap_values, axis = 0)), axis = 0)
    importances = np.abs(np.dot(importances_pca, pca.components_))
    #std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    var_sorted = features[indices]
    var_imp_sorted = importances[indices]

    plt.rcParams["figure.figsize"] = (10, num_features/2.5) # (w, h)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.barh(var_sorted[:num_features],
            var_imp_sorted[:num_features], color = 'blue',
            #yerr=std[indices][:num_features], ecolor='black',
            align="center")
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title('Feature Importances for Whole Spectrum')
    plt.tight_layout()
    plt.savefig(dir_out + 'whole' + '/MLP_Importances_whole' + structure +'.png')
    plt.close()