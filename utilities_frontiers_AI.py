# This files contains all funtions and classes needed for running the code in test_notebook.ipynb

import os 
import errno
from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import RepeatedKFold
import warnings
import string
import math
import random
import seaborn as sns


seed = 1

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


##################
####FUNCTIONS#####
##################

def prepareDataAndTrainModel(device, filesTraining, filesPrediction):
    """Execute different analysis and plots on the test dataset
        
    Parameters
    ----------  
    device : string
        The string of the device to be used to train the model
    filesTraining : list
        The files of training data
    filesPrediction : list
        The files of prediction data

    Returns
    -------
    analysis : analysis
        The analysis object
    mse_test : float
        The MSE of the test set
    mae_test : float
        The MAE of the test set
    """    
    
    #Prepare training data
    preparedData = getPreparedDataFromFiles(filesTraining, augmentation=False)
    #Prepare test data
    preparedDataPrediction = getPreparedDataFromFiles(filesPrediction, test_size=0, scaler_x=preparedData.getScalerX(), scaler_y=preparedData.getScalerY())
    #Train the model
    with tf.device(device):   
        preparedModel = PrepareModel(preparedData.getDSTrain(), preparedData.getDSTest())
    #Plot loss history
    preparedModel.plotLossHistory()
    return preparedData, preparedDataPrediction, preparedModel

def analysisTestDataset(preparedModel, preparedData, preparedDataPrediction, color):
    """Execute different analysis and plots on the test dataset
        
    Parameters
    ----------  
    preparedModel : preparedModel 
        The model used for prediction
    preparedData : preparedData
        The validation data
    preparedDataPrediction : preparedData
        The test data
    color : color
        The color used to plot the predicted data

    Returns
    -------
    analysis : analysis
        The analysis object
    mse_test : float
        The MSE of the test set
    mae_test : float
        The MAE of the test set
    """    
    
    #Create analysis object and calculate MSE
    analysis = Analysis(preparedModel, preparedData.getScalerY(), ds=preparedDataPrediction.getDSTrain())
    _ = analysis.calculateError()
    mse_test = analysis.calculateError(labelNr=1)
    mae_test = analysis.calculateError(labelNr=1, mode="mae")
    _ = analysis.calculateError(scaled=False)

    #Plot an excerpt of ground truth and prediction
    xlim_from1 = 1300
    xlim_to1 = 1800
    analysis.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from1, xlim_to=xlim_to1, color=color)

    #Plot an excerpt of ground truth and prediction
    xlim_from2 = 2000
    xlim_to2 = 2500
    analysis.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from2, xlim_to=xlim_to2, color=color)

    #Scatter plot of prediction
    analysis.scatterPrediction(1, lim_from=0, lim_to=0.6)

    #Plot permutation feature importance
    analysis.showFeatureImportance(preparedModel, preparedDataPrediction.getDSTrain(asNumpy=True), preparedDataPrediction.getFeatures(), title="random tracks", color=color)

    #Visualize the track ground truth, prediction and error
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="ground truth")
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="prediction")
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="error")

    #Visualize the track ground truth, prediction and error and show also rectangles indicating the excerpts
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="ground truth", excerpts=[[xlim_from1, xlim_to1], [xlim_from2, xlim_to2]])
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="prediction", excerpts=[[xlim_from1, xlim_to1], [xlim_from2, xlim_to2]])
    analysis.visualizeTrack(preparedDataPrediction, 1, preparedDataPrediction.getTracks(), mode="error", excerpts=[[xlim_from1, xlim_to1], [xlim_from2, xlim_to2]])

    return analysis, mse_test, mae_test

def analysisValidationDataset(preparedModel, preparedData, color, filesSingleTracks):
    """Execute different analysis and plots on the validation dataset
        
    Parameters
    ----------  
    preparedModel : preparedModel 
        The model used for prediction
    preparedData : preparedData
        The validation data
    color : color
        The color used to plot the predicted data
    filesSingleTracks : list
        The files of the single tracks

    Returns
    -------
    analysis : analysis
        The analysis object
    mse_val : float
        The MSE of the validation set
    mae_val : float
        The MAE of the validation set
    """    
    
    #Create analysis object and calculate MSE
    analysis = Analysis(preparedModel, preparedData.getScalerY(), ds=preparedData.getDSTest())
    _ = analysis.calculateError()
    mse_val = analysis.calculateError(labelNr=1)
    mae_val = analysis.calculateError(labelNr=1, mode="mae")
    _ = analysis.calculateError(scaled=False)

    #Single track #18
    analysis.visualizePrediction(1, "Single track #18", xlim_from=313, xlim_to=545, color=color)

    #V-track #41
    analysis.visualizePrediction(1, "V-track #41", xlim_from=3478, xlim_to=3988, color=color)

    #Spiral track left
    analysis.visualizePrediction(1, "Spiral track left (excerpt)", xlim_from=23000, xlim_to=25000, color=color)

    #Scatter plot of prediction
    analysis.scatterPrediction(1, lim_from=0, lim_to=0.6)

    #Plot permutation feature importance
    analysis.showFeatureImportance(preparedModel, preparedData.getDSTest(asNumpy=True), preparedData.getFeatures(), title="simple geometries", color=color)

    #Single Track: Visualize the track ground truth, prediction and error
    analysis.visualizeTrack(preparedData, 1, [4], mode="ground truth", title="Single track #")
    analysis.visualizeTrack(preparedData, 1, [4], mode="prediction", title="Single track #")
    analysis.visualizeTrack(preparedData, 1, [4], mode="error", title="Single track #")

    #V-Track: Visualize the track ground truth, prediction and error
    analysis.visualizeTrack(preparedData, 1, [41], mode="ground truth", title="V-track #")
    analysis.visualizeTrack(preparedData, 1, [41], mode="prediction", title="V-track #")
    analysis.visualizeTrack(preparedData, 1, [41], mode="error", title="V-track #")

    #Spiral Track: Visualize the track ground truth, prediction and error
    analysis.visualizeTrack(preparedData, 1, [66], mode="ground truth", title="Spiral track #")
    analysis.visualizeTrack(preparedData, 1, [66], mode="prediction", title="Spiral track #")
    analysis.visualizeTrack(preparedData, 1, [66], mode="error", title="Spiral track #")

    preparedData_single = getPreparedDataFromFiles(filesSingleTracks, test_size=0, scaler_x=preparedData.getScalerX(), scaler_y=preparedData.getScalerY())
    analysis.plotHvsVP(preparedData_single)

    return analysis, mse_val, mae_val

def compareTrainValidationAndTest(analysis, preparedData, preparedDataPrediction, preparedModel, mse_val, mae_val, mse_test, mae_test):
    """Compare different datasets and the data distribution and print a summary
        
    Parameters
    ----------  
    analysis : analysis 
        Analysis object of experiment
    preparedData : preparedData 
        The train and validation data
    preparedDataPrediction : preparedData
        The test data
    preparedModel : preparedModel
        The model to be used for prediction
    mse_val : float
        The MSE of the validation set
    mae_val : float
        The MAE of the validation set
    mse_test : float
        The MSE of the test set
    mae_test : float
        The MAE of the test set
    """

    #Show the feature and label distribution
    analysis.compareFeaturesAndTargets(preparedData.getData(), preparedData.getTarget(), preparedDataPrediction.getData(), preparedDataPrediction.getTarget(), preparedData.getFeatures(), preparedData.getLabels())

    #Calculate MSE/MAE of single tracks
    excludeTracks = np.append(0, preparedData.getTracks(scope="train"))
    excludeTracks.sort()
    files = [["SingleTracks_20191008", "/imageData_20191008.csv", "/trackData_20191008.csv", excludeTracks, 0.032]]
    preparedData_single = getPreparedDataFromFiles(files, test_size=0, scaler_x=preparedData.getScalerX(), scaler_y=preparedData.getScalerY())
    analysis = Analysis(preparedModel, preparedData_single.getScalerY(), ds=preparedData_single.getDSTrain())
    _ = analysis.calculateError()
    mse_single = analysis.calculateError(labelNr=1, mode="mse")
    mae_single = analysis.calculateError(labelNr=1, mode="mae")
    _ = analysis.calculateError(scaled=False)

    #Calculate MSE/MAE of V tracks
    excludeTracks = np.append(37, preparedData.getTracks(scope="train"))-37
    excludeTracks.sort()
    files = [["V_tracks", "/imAll.csv", "/trAll.csv", excludeTracks, 0.099]]
    preparedData_v = getPreparedDataFromFiles(files, test_size=0, scaler_x=preparedData.getScalerX(), scaler_y=preparedData.getScalerY(), start_trackNr=37)
    analysis = Analysis(preparedModel, preparedData_v.getScalerY(), ds=preparedData_v.getDSTrain())
    _ = analysis.calculateError()
    mse_v = analysis.calculateError(labelNr=1, mode="mse")
    mae_v = analysis.calculateError(labelNr=1, mode="mae")
    _ = analysis.calculateError(scaled=False)

    #Calculate MSE/MAE of spiral tracks
    files = [["Spiral_tracks", "/imData_20200731_left.csv", "/trackData_20200731_left.csv", [0], 0.0825]]
    preparedData_spiral = getPreparedDataFromFiles(files, test_size=0, scaler_x=preparedData.getScalerX(), scaler_y=preparedData.getScalerY(), start_trackNr=37)
    analysis = Analysis(preparedModel, preparedData_spiral.getScalerY(), ds=preparedData_spiral.getDSTrain())
    _ = analysis.calculateError()
    mse_spiral = analysis.calculateError(labelNr=1, mode="mse")
    mae_spiral = analysis.calculateError(labelNr=1, mode="mae")
    _ = analysis.calculateError(scaled=False)

    #Calculate the hight means
    idx = preparedData.getData()[preparedData.getData()["track#"].isin(preparedData.getTracks(scope="test"))]
    h_val = preparedData.getTarget().iloc[idx.index].H.mean()
    h_single = preparedData_single.getTarget().H.mean()
    h_v = preparedData_v.getTarget().H.mean()
    h_spiral = preparedData_spiral.getTarget().H.mean()
    h_test = preparedDataPrediction.getTarget().H.mean()

    #Print a summary
    summary = { "validation": [mse_val, mae_val, h_val, 100*mae_val/h_val],
            "single": [mse_single, mae_single, h_single, 100*mae_single/h_single],
            "v-track": [mse_v, mae_v, h_v, 100*mae_v/h_v],
            "spiral": [mse_spiral, mae_spiral, h_spiral, 100*mae_spiral/h_spiral],
            "random": [mse_test, mae_test, h_test, 100*mae_test/h_test]
    }
    print ("{:<15} {:<10} {:<10} {:<10} {:<10}".format('Geometry','MSE', 'MAE', '<H>','Error %'))
    for k, v in summary.items():
        mse, mae, h, perc = v
        print ("{:<15} {:<10} {:<10} {:<10} {:<10}".format(k, mse.round(5), mae.round(5), h.round(3), perc.round(2)))

def compareBothExperiments(analysis_ex1p, analysis_ex2p, color_ex1, color_ex2):
    """Compare both experiments and plot excerpts for the test set
        
    Parameters
    ----------  
    analysis_ex1p : analysis 
        Analysis object of experiment 1
    analysis_ex2p : analysis 
        Analysis object of experiment 2
    color_ex1 : color
        The color used to plot prediction of experiment 1
    color_ex2 : color
        The color used to plot prediction of experiment 2
    """    

    #Plot an excerpt of ground truth and prediction for both experiments
    xlim_from1 = 1300
    xlim_to1 = 1800
    plt.figure(figsize=(15,5))
    analysis_ex1p.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from1, xlim_to=xlim_to1, color=color_ex1, label="LSTM experiment 1", show=False, gt=False)
    analysis_ex2p.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from1, xlim_to=xlim_to1, color=color_ex2, label="LSTM experiment 2", show=False)
    plt.show()

    #Plot an excerpt of ground truth and prediction for both experiments
    xlim_from2 = 2000
    xlim_to2 = 2500
    plt.figure(figsize=(15,5))
    analysis_ex1p.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from2, xlim_to=xlim_to2, color=color_ex1, label="LSTM experiment 1", show=False, gt=False)
    analysis_ex2p.visualizePrediction(1, "Random track #1 (excerpt)", xlim_from = xlim_from2, xlim_to=xlim_to2, color=color_ex2, label="LSTM experiment 2", show=False)
    plt.show()

def getPreparedDataFromFiles(files, test_size=0.25, scaler_x=None, scaler_y=None, augmentation=False, start_trackNr=0, printout=True):
    """Returns a preparedData object containing the data for the given files
        
    Parameters
    ----------  
    files : list 
        List of files for each given following: run/imgfile/trackfile/drop_tracks
    test_size : decimal, optional
        Part of the data used as test data (default = 0.25)
    scaler_x : Scaler, optional
        The scaler used for the features X
    scaler_y : Scaler, optional
        The scaler used for the labels y
    augmentation : boolean, optional
        Defines if a data aufmentation (rotation, mirroring) have to take place (default = False)
    start_trackNr : int, optional
        Defines the starting number for all tracks
    printout : boolean, optional
        Defines if statistics have to be printed out

    Returns
    -------
    PrepareData
        The preparedData object
    """    

    preparedData = None
    for run, imgfile, trkfile, drop_tracks, powder_flux in files:
        if printout:
            print("---------------------------------------------------")
            print("Run: {0}".format(run))    
        pdRun = PrepareData(run=run, imgFile=imgfile, trkFile=trkfile, keep_tracks=True, kindOfScale='MinMax', drop_columns=["I_mean_movmean", "I_mean_crop", "I_mean_crop_movmean", "I_std"], drop_tracks=[t+start_trackNr for t in drop_tracks], start_trackNr=start_trackNr, test_size=test_size, scaler_x=scaler_x, scaler_y=scaler_y, augmentation=augmentation, powder_flux=powder_flux)
        if printout:
            print("Features: {0}".format(pdRun.getFeatures()))
            print("Labels: {0}".format(pdRun.getLabels()))
            print("Train tracks: {0}".format(pdRun.getTracks(scope="train")))
            print("Test tracks: {0}".format(pdRun.getTracks(scope="test")))
            print("Tracks: {0}".format(pdRun.getTracks()))
        start_trackNr = pdRun.getTracks().max() + 1
        if preparedData == None:
            preparedData = pdRun
            scaler_x = pdRun.getScalerX()
            scaler_y = pdRun.getScalerY()
        else:
            preparedData.merge(pdRun)
    if printout:
        print("==================================================")
        print("Final prepared data...")   
        print("Train tracks: {0}".format(preparedData.getTracks(scope="train")))
        print("Test tracks: {0}".format(preparedData.getTracks(scope="test")))
        print("Tracks: {0}".format(preparedData.getTracks()))
    return preparedData

def setPltParams():
    """Sets the matplotlib parameters

    """  
    
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.which"] = "both"
    plt.rcParams["axes.grid.axis"] = "both"
    plt.rcParams["axes.facecolor"] = "whitesmoke"
    plt.rcParams["axes.titlesize"] = "20"
    plt.rcParams["axes.labelsize"] = "18"
    plt.rcParams["axes.edgecolor"] = "lightgrey"
    plt.rcParams["grid.color"] = "lightgrey"
    plt.rcParams["boxplot.whiskerprops.linewidth"] = 2.0
    plt.rcParams["boxplot.boxprops.linewidth"] = 2.0
    plt.rcParams["boxplot.capprops.linewidth"] = 2.0
    plt.rcParams["ytick.labelsize"] = "14"
    plt.rcParams["xtick.labelsize"] = "14"
    plt.rcParams["legend.fontsize"] = "18"
    plt.rcParams["figure.titlesize"] = "25"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


##################
#####CLASSES######
##################

class DatasetGenerator():
    """This class generates time series tensorflow BatchDatasets grouped by track# with the given time window

    Attributes
    ----------
    features : dataframe
        Containing the features X
    labels : dataframe
        Containing the labels y
    time_steps : int
        Determines the window size of the timeseries
    sampling_rate : int, optional
        Determines the sampling rate of the timeseries
    batch_size : int, optional
        Determines the batch size of the timeseries (default is 128)   

    Methods
    -------
    getDataSet() 
        Returns a Tensorflow Dataset containing the features and labels with the given time window (time_steps)

    """    

    def __init__(self, features, labels, time_steps, sampling_rate=1, batch_size=128):
        self.DataSets = list()
        self.trackidx = list()
        for track in features["track#"].unique().astype("int32"):
            if "augmentation" in features.columns:
                for aug in features["augmentation"].unique():
                    X = features[features['track#'] == track].copy()
                    X = X[X["augmentation"]==aug]
                    X.drop(["track#", "augmentation"], axis=1, inplace=True)
                    y = labels[features['track#'] == track].copy()
                    ds = tf.keras.preprocessing.timeseries_dataset_from_array(X.values, np.roll(y.values, -time_steps, axis=0), time_steps, sampling_rate = sampling_rate, batch_size = batch_size)
                    self.DataSets.append(ds)
                    self.trackidx.append(np.repeat(track, y.shape[0]-time_steps))
            else:
                X = features[features['track#'] == track].copy()
                X.drop("track#", axis=1, inplace=True)
                y = labels[features['track#'] == track].copy()
                ds = tf.keras.preprocessing.timeseries_dataset_from_array(X.values, np.roll(y.values, -time_steps, axis=0), time_steps, sampling_rate = sampling_rate, batch_size = batch_size)
                self.DataSets.append(ds)
                self.trackidx.append(np.repeat(track, y.shape[0]-time_steps+1))

    def getDataSet(self, asNumpy=False):
        """Returns a Tensorflow Dataset containing the stacked features and labels of all tracks considering the given timeseries window (time_steps)
        
        Parameters
        ----------  
        asNumpy : boolean, optional 
            Defines if the dataset should be converted into a numpy array (default is false)

        Returns
        -------
        Dataset
            Timeseries containing stacked features and labels for all the tracks
        """
        if len(self.DataSets) > 0:
            ds = self.DataSets[0]
            for i in range(1, len(self.DataSets)): 
                ds = ds.concatenate(self.DataSets[i])
            if asNumpy:
                ds = np.asarray(list(ds.as_numpy_iterator()))
            return ds, np.concatenate(self.trackidx)
        else:
            return None, []

class PrepareData:
    """This class reads the image and track data and perform the preprocessing in order to use the data for model training

    Attributes
    ----------
    run : str, optional
        Folder name of the track run (default is SingleTracks_20191008)
    imgFile : str, optional
        File name containing the image data (default is /imageData_20191008.csv)
    trkFile : str, optional
        File name containing the track data (default is /trackData_20191008.csv) 
    batch_size : int, optional
        Determines the batch size of the timeseries (default is 64)
    time_steps : int, optional
        Determines the window size of the timeseries (default is 20)
    kindOfScale : str, optional
        Determine the type of scaler to be used (default is 'std'):
        - 'std' : StandardScaler()
        - 'MinMax' : MinMaxScaler()
        - 'robust' : RobustScaler()
        - 'quantile' : QuantileTransformer()
    kindOfSplit : str, optional
        Defines the kind of train and test split (default is 'random'):
        - random: the track# to split into train and test will be chosen randomly
        - given: the track# to split into train and test will be given by the parameters given_train_tracks and given_test_tracks
    given_train_tracks : str, optional
        List of track# to be used as train set if kindOfSplit='given' (default is None)
    given_test_tracks : str, optional
        List of track# to be used as test set if kindOfSplit='given' (default is None)
    test_size : decimal, optional
        If kindOfSplit='random' this parameter defines the ratio for splitting into train and test (default is 0.25 which means 25% will be used for test)
    keep_tracks : boolean, optional
        Defines if the column 'track#' should be kept or dropped (default is False)
    drop_columns : list, optional
        Defines the column names of the data to be dropped (default is None)
    drop_tracks : list, optional
        Defines the tracks not to be considered
    acc_calculation: boolean, optional
        Defines if the calculation of acceleration/deceleration/stable have to take place
    acc_window : int, optional
        The window size to be considered to determine gradient of velocity (default is 4)
    acc_thresholdGradient : int, optional
        Only if the velocity change within the defined window is above this threshold, it will be considered as acceleration or deceleration (default is 20/60000)
    acc_filterG1Zero : boolean, optional
        Determines if the calculation have only to take place in case of G1=1 (default is True)     
    speed_vector : boolean, optional
        Determines if the speed vector have to be added as feature (default is False)    
    start_trackNr : int, optional
        Determines the number of the first track (default = 0)
    scaler_x : Scaler, optional
        The scaler used for the features X
    scaler_y : Scaler, optional
        The scaler used for the labels y
    augmentation : boolean, optional
        Defines if a data aufmentation (rotation, mirroring) have to take place (default = False)
    powder_flux : float, optional
        The amount of powder in g/s (default is 0)

    Methods
    -------
    getDSTrain()
        Returns the train timeseries
    getDSTest()
        Returns the test timeseries
    getScalerX()
        Returns the Scaler used for features X
    getScalerY()
        Returns the Scaler used for labels y
    def getData()
        Returns the data of the tracks        
    def getTarget()
        Returns the target of the tracks   
    def getTracks()
        Returns the track#s       
    def getTime()
        Returns the times for the tracking data
    def getFeatures()
        Returns the list of features
    def getLabels()
        Returns the list of labels
    def getXYPos()
        Returns the X, Y positions of the tracks
    def merge()
        Merge the current PrepareData object with the data of a given PrepareData object
    """

    path = "./Data/"

    def __init__(self, 
                 run = "SingleTracks_20191008", 
                 imgFile = "/imageData_20191008.csv", 
                 trkFile = "/trackData_20191008.csv", 
                 batch_size=64, 
                 time_steps=20, 
                 kindOfScale='MinMax', 
                 kindOfSplit='random', 
                 given_train_tracks=None, 
                 given_test_tracks=None, 
                 test_size=0.25, 
                 keep_tracks=False,
                 drop_columns=None, 
                 drop_tracks=None,
                 acc_calculation=False,
                 acc_window=4, 
                 acc_thresholdGradient=20/60000, 
                 acc_filterG1Zero=True,
                 speed_vector=True,
                 start_trackNr=0,
                 scaler_x=None,
                 scaler_y=None,
                 augmentation=False,
                 powder_flux=0):

        self.run = run
        self.imgFile = imgFile
        self.trkFile = trkFile
        self.time_steps = time_steps

        self.ds_train, self.ds_test, self.scaler_x, self.scaler_y, self.data, self.target, self.tracks, self.time, self.features, self.labels, self.XYPos, self.train_tracks, self.test_tracks, self.trackidx_train, self.trackidx_test =  \
                self.__getTrainAndTestTimeseries(   batch_size=batch_size, 
                                                    time_steps=time_steps,
                                                    kindOfScale=kindOfScale,
                                                    kindOfSplit=kindOfSplit,
                                                    given_train_tracks=given_train_tracks,
                                                    given_test_tracks=given_test_tracks,
                                                    test_size=test_size,
                                                    keep_tracks=keep_tracks,
                                                    drop_columns=drop_columns,
                                                    drop_tracks=drop_tracks,
                                                    acc_calculation=acc_calculation,
                                                    acc_window=acc_window,
                                                    acc_thresholdGradient=acc_thresholdGradient,
                                                    acc_filterG1Zero=acc_filterG1Zero,
                                                    speed_vector=speed_vector,
                                                    start_trackNr=start_trackNr,
                                                    scaler_x=scaler_x,
                                                    scaler_y=scaler_y,
                                                    augmentation=augmentation,
                                                    powder_flux=powder_flux)

    def getDSTrain(self, asNumpy=False):
        """Returns the train timeseries

        Parameters
        ----------  
        asNumpy : boolean, optional 
            Defines if the dataset should be converted into a numpy array (default is false)

        Returns
        -------
        timeseries
            containing the training data
        """     
        ds = self.ds_train
        if asNumpy:
            ds = np.asarray(list(ds.as_numpy_iterator()))
        return ds    

    def getDSTest(self, asNumpy=False):
        """Returns the test timeseries

        Parameters
        ----------  
        asNumpy : boolean, optional 
            Defines if the dataset should be converted into a numpy array (default is false)

        Returns
        -------
        timeseries
            containing the test data
        """     
        ds = self.ds_test
        if asNumpy:
            ds = np.asarray(list(ds.as_numpy_iterator()))
        return ds  

    def getScalerX(self):
        """Returns the Scaler used for features X

        Returns
        -------
        Scaler
            The scaler used for the features X
        """     
        return self.scaler_x    

    def getScalerY(self):
        """Returns the Scaler used for labels y

        Returns
        -------
        Scaler
            The scaler used for the labels y
        """     
        return self.scaler_y          

    def getData(self, windowed=False, trackNr=None):
        """Returns the data of the tracks

        Parameters
        ----------
        windowed : boolean, optional
            Determines if the whole data or only the windowed part (skip first timesteps) should be returned (default is False)
        trackNr : int, optional
            Defines the track for which the data should be returned (default is None which means all data are returned)

        Returns
        -------
        array
            The data of the tracks
        """     
        if trackNr is not None:
            d = self.data[self.data["track#"] == trackNr]
        else:
            d = self.data
        if windowed:
            return d.loc[d["track#"] == d["track#"].shift(19)]
        else:
            return d  

    def getTarget(self):
        """Returns the target of the tracks

        Returns
        -------
        array
            The targets of the tracks
        """     
        return self.target           

    def getTracks(self, scope="all"):
        """Returns the track#s

        Parameters
        ----------
        scope : string, optional
            Which track# to be returned ['all', 'train', 'test'] (default is all)
  
        Returns
        -------
        int
            The list of track#
        """     
        if scope=="all":
            return np.sort(self.tracks)
        elif scope=="train":
            return np.sort(self.train_tracks)
        elif scope=="test":
            return np.sort(self.test_tracks)

    def getTime(self, windowed=False):
        """Returns the times for the tracking data
  
        Parameters
        ----------
        windowed : boolean, optional
            Determines if the whole data or only the windowed part (skip first timesteps) should be returned (default is False)

        Returns
        -------
        dataframe
            The list of times
        """     
        if windowed:
            return self.time.loc[self.data["track#"] == self.data["track#"].shift(19)]
        else:
            return self.time        

    def getFeatures(self):
        """Returns the list of features
  
        Returns
        -------
        list
            The list of features
        """     
        return self.features   

    def getLabels(self):
        """Returns the list of labels
  
        Returns
        -------
        list
            The list of labels
        """     
        return self.labels       

    def getXYPos(self):
        """Returns the X, Y positions of the tracks
  
        Returns
        -------
        list
            The list of X, Y positions
        """     
        return self.XYPos    

    def merge(self, preparedData):
        """Merge the current PrepareData object with the data of a given PrepareData object
  
        Parameters
        ----------
        preparedData : PrepareData
            The PrepareData object to be merged with the current PrepareData object

        Raises
        ------
        ValueError
            The features and/or labels of the given PrepareData object does not match to the current object!
        """     
        if self.features == preparedData.getFeatures() and self.labels == preparedData.getLabels():
            if preparedData.getDSTrain() != None:
                self.ds_train = self.ds_train.concatenate(preparedData.getDSTrain())
            if preparedData.getDSTest() != None:
                self.ds_test = self.ds_test.concatenate(preparedData.getDSTest())
            if not preparedData.getData().empty:
                self.data = pd.concat([self.data, preparedData.getData()])
                self.data.reset_index(drop=True, inplace=True)
            if not preparedData.getTarget().empty:
                self.target = pd.concat([self.target, preparedData.getTarget()])
                self.target.reset_index(drop=True, inplace=True)
            if preparedData.getTracks().any():
                self.tracks = np.append(self.tracks, preparedData.getTracks())
            if preparedData.getTime().any():
                self.time = np.append(self.time, preparedData.getTime())
            if not preparedData.getXYPos().empty:
                self.XYPos = pd.concat([self.XYPos, preparedData.getXYPos()])
                self.XYPos.reset_index(drop=True, inplace=True)
            if len(preparedData.getTracks(scope="train")) > 0:
                self.train_tracks = np.append(self.train_tracks, preparedData.getTracks(scope="train"))
            if len(preparedData.getTracks(scope="test")) > 0:
                self.test_tracks = np.append(self.test_tracks, preparedData.getTracks(scope="test"))
            if len(preparedData.trackidx_train) > 0:
                self.trackidx_train = np.append(self.trackidx_train, preparedData.trackidx_train)
            if len(preparedData.trackidx_test) > 0:
                self.trackidx_test = np.append(self.trackidx_test, preparedData.trackidx_test)
        else:
            raise ValueError("The features and/or labels of the given PrepareData object does not match to the current object!")

    def __readCSV(self):
        """Reads the both csv files containing the image and track data

        Returns
        -------
        dataframe
            containing the image data
        dataframe
            containing the track data

        Raises
        ------
        FileNotFoundError
            Raised if image or track file path not exist
        """    

        #Read the image csv file
        if isinstance(self.imgFile, list):
            imgData = None
            for imgFile in self.imgFile:
                imgPath = self.path + self.run + imgFile
                if os.path.exists(imgPath):
                    if imgData is None:
                        imgData = pd.read_csv(imgPath, sep=";")
                    else:
                        imgData = pd.concat([imgData, pd.read_csv(imgPath, sep=";")])
                        imgData.fillna(value=0, inplace=True)
                        imgData.reset_index(drop=True, inplace=True)
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), imgPath)
        else:
            imgPath = self.path + self.run + self.imgFile
            if os.path.exists(imgPath):
                imgData = pd.read_csv(imgPath, sep=";")
                imgData.fillna(value=0, inplace=True)
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), imgPath)

        #Read the track csv file
        if isinstance(self.trkFile, list):
            trkData = None
            for trkFile in self.trkFile:
                trkPath = self.path + self.run + trkFile
                if os.path.exists(trkPath):
                    if trkData is None:
                        trkData = pd.read_csv(trkPath, sep=";")
                    else:
                        trkData = pd.concat([trkData, pd.read_csv(trkPath, sep=";")], axis=0)
                        trkData.fillna(value=0, inplace=True)
                        trkData.reset_index(drop=True, inplace=True)
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), trkPath)
        else:
            trkPath = self.path + self.run + self.trkFile
            if os.path.exists(trkPath):
                trkData = pd.read_csv(trkPath, sep=";")
                trkData.fillna(value=0, inplace=True)
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), trkPath)

        return imgData, trkData

    def __data_preparation(self, imgData, trkData, drop_tracks=None, speed_vector=False, acc_calculation=True, acc_window=4, acc_thresholdGradient=20, acc_filterG1Zero=True, start_trackNr=0, addHist=True, augmentation=False, powder_flux=0):
        """Prepare the image and track data and returns the features and labels

        Parameters
        ----------
        imgData : dataframe
            Dataframe containing the image data
        trkData : dataframe
            Dataframe containing the track data
        drop_tracks : list, optional
            Defines the tracks not to be considered            
        speed_vector : boolean, optional
            Determines if the speed vector have to be added as feature (default is False)
        acc_window : int, optional
            The window size to be considered to determine gradient of velocity (default is 4)
        acc_thresholdGradient : int, optional
            Only if the velocity change within the defined window is above this threshold, it will be considered as acceleration or deceleration (default is 20)
        acc_filterG1Zero : boolean, optional
            Determines if the calculation have only to take place in case of G1=1 (default is True)
        start_trackNr : int, optional
            Determines the number of the first track (default = 0)
        addHist : boolean, optional
            Defines if the count of planned trajectory steps has to be considered
        augmentation : boolean, optional
            Defines if a data aufmentation (rotation, mirroring) have to take place (default = False)
        powder_flux : float, optional
            The amount of powder in g/s (default is 0)

        Returns
        -------
        dataframe
            containing the features/data
        dataframe
            containing the labels/target
        list
            The list of tracks
        list
            The list of timestamps
        list
            The list of X, Y positions
        """

        #roundig the data
        imgData = imgData.round({'I_mean':2, 'I_mean_crop':2, 'I_median':2, 'I_median_crop':2, 'I_mean_movmean':2, 'I_mean_crop_movmean':2})
        trkData = trkData.round({'D':3, 'H':3, 'A':7})
        trkData.drop('sdres', axis=1, inplace=True)
        
        #set targets and data
        target = trkData[['D', 'H', 'A']].copy()
        data = imgData.copy()
        data["powder_flux"] = powder_flux
        data['V'] = trkData['V'] / 60000
        data['G1'] = trkData['G1']
        data["Pnom"] = trkData["Pnom"]
        data["Xpos"] = trkData["Xpos"]
        data["Ypos"] = trkData["Ypos"]

        #add track number
        data, tracks = self.__add_track_number(data, start_trackNr)

        #drop tracks
        if drop_tracks is not None:
            for drop in drop_tracks:
                idx = data[data["track#"] == drop].index
                target.drop(labels=idx, inplace=True)  
                data.drop(labels=idx, inplace=True)  
                trkData.drop(labels=idx, inplace=True)  
                tracks = np.delete(tracks, np.where(tracks == drop))

        #Determine the count of laser occurances in a given distance
        if addHist:
            def getHistInDist(hist, x, thdist, mean_t):
                hist["dist"] = np.sqrt( (hist["Xpos"]-x["Xpos"])**2 + (hist["Ypos"]-x["Ypos"])**2 )
                return hist[hist["dist"] <= thdist].shape[0] / mean_t
            mean_t = 5
            mean_D = 1
            data['points_D'] = data.apply(lambda x: getHistInDist(data[data["track#"] == x["track#"]].copy(), x, mean_D, mean_t), axis=1)
            data['points_D/2'] = data.apply(lambda x: getHistInDist(data[data["track#"] == x["track#"]].copy(), x, mean_D/2, mean_t), axis=1)
            
        #Data augmentation
        if augmentation:
            aug = 0
            data_a = data.copy()
            data_a['augmentation'] = aug
            target_a = target.copy()
            #Rotation
            for i in range(10):
                aug += 1
                angle =  np.radians(np.random.uniform(1, 180) + data["track#"])
                data_c = data.copy()
                data_c["Xpos"] = (data["Xpos"] * np.cos(angle)) - (data["Ypos"] * np.sin(angle))
                data_c["Ypos"] = (data["Xpos"] * np.sin(angle)) + (data["Ypos"] * np.cos(angle))
                data_c["augmentation"] = aug
                data_a = pd.concat([data_a, data_c], ignore_index=True)
                target_a = pd.concat([target_a, target.copy()], ignore_index=True)
            #Mirroring
            data_c = data_a.copy()
            data_c["Xpos"] = data_a["Ypos"]
            data_c["Ypos"] = data_a["Xpos"]
            data_c["augmentation"] = data_c["augmentation"] + aug + 1
            data_a = pd.concat([data_a, data_c], ignore_index=True)
            target_a = pd.concat([target_a, target_a.copy()], ignore_index=True)
            data = data_a  
            target = target_a

        # Add speed vector if needed
        if speed_vector:
            #Don't determine vector for each new track or augmentation 
            if augmentation:
                consider = ((data["track#"].diff() == 0) & (data["augmentation"].diff() == 0))
            else:
                consider = (data["track#"].diff() == 0)
            dt = data["t"].diff()
            data["V_x"] = np.abs(consider * data["Xpos"].diff()) / dt
            data['V_x'].fillna(value=0, inplace=True)
            data["V_y"] = np.abs(consider * data["Ypos"].diff()) / dt
            data['V_y'].fillna(value=0, inplace=True)
        
        time = data['t'].copy()
        XYPos = trkData[["Xpos", "Ypos"]].copy()
        data.drop(['laserON', 'img_name', 't', "Xpos", "Ypos"], axis=1, inplace=True)

        #add accelerating, decelerating, stable
        if acc_calculation:
            data = self.__add_accel_decel_stable(data, tracks, acc_window=acc_window, acc_thresholdGradient=acc_thresholdGradient, acc_filterG1Zero=acc_filterG1Zero)
        
        return data, target, tracks, time, XYPos

    def __add_track_number(self, data, start_trackNr=0):
        """Add the track number to the data dataframe

        Parameters
        ----------
        data : dataframe
            Dataframe containing the feature data
        start_trackNr : int, optional
            Determines the number of the first track (default = 0)

        Returns
        -------
        dataframe
            Dataframe containing the feature data with added track number
        """        

        #add track number info
        data['start'] = ((data['G1'] != data['G1'].shift(1)) & (data['G1']==1)).astype(int).cumsum()
        data['track#'] = data["start"].loc[data["G1"]==1] + start_trackNr
        data['track#'].fillna(value=start_trackNr, inplace=True)
        data.drop('start', axis=1, inplace=True)
        return data, data["track#"].unique().astype(int).flatten()

    def __add_accel_decel_stable(self, data, acc_window=4, acc_thresholdGradient=20 / 60000, acc_filterG1Zero=True):
        """Add the acceleration/deceleration/stable one-hot encoded columns to the features dataframe

        Parameters
        ----------
        data : dataframe
            Dataframe containing the feature data
        acc_window : int, optional
            The window size to be considered to determine gradient of velocity (default is 4)
        acc_thresholdGradient : int, optional
            Only if the velocity change within the defined window is above this threshold, it will be considered as acceleration or deceleration (default is 20)
        acc_filterG1Zero : boolean, optional
            Determines if the calculation have only to take place in case of G1=1 (default is True) 

        Returns
        -------
        dataframe
            Dataframe containing the feature data with added acceleration/deceleration/stable one-hot encoded columns
        """        

        #Add accelerating, decelerationg, stable info
        if acc_filterG1Zero:
            consider = (data["G1"] != 0).astype(int)
        else:
            consider = 1

        ma = data['V'].rolling(acc_window, min_periods=1).mean()
        data["decelerating"] = (data['V'] - ma < - acc_thresholdGradient).astype(int) * consider
        data["accelerating"] = (data['V'] - ma > acc_thresholdGradient).astype(int) * consider
        data["stable"] = (data["accelerating"] + data["decelerating"] == 0).astype(int) * consider            

        return data

    def __getDataAndTarget(self, drop_tracks=None, acc_calculation=True, acc_window=4, acc_thresholdGradient=20, acc_filterG1Zero=True, speed_vector=False, start_trackNr=0, augmentation=False, powder_flux=0):
        """Returns the data and target dataframes

        Parameters
        ----------
        drop_tracks : list, optional
            Defines the tracks not to be considered     
        acc_calculation: boolean, optional
            Defines if the calculation of acceleration/deceleration/stable have to take place   
        acc_window : int, optional
            The window size to be considered to determine gradient of velocity (default is 4)
        acc_thresholdGradient : int, optional
            Only if the velocity change within the defined window is above this threshold, it will be considered as acceleration or deceleration (default is 20)
        acc_filterG1Zero : boolean, optional
            Determines if the calculation have only to take place in case of G1=1 (default is True)
        speed_vector : boolean, optional
                    Determines if the speed vector have to be added as feature (default is False)
        start_trackNr : int, optional
            Determines the number of the first track (default = 0) 
        augmentation : boolean, optional
            Defines if a data aufmentation (rotation, mirroring) have to take place (default = False) 
        powder_flux : float, optional
            The amount of powder in g/s (default is 0)

        Returns
        -------
        dataframe
            containing the data
        dataframe
            containing the target
        list
            The list of tracks
        list
            The list of timestamps
        list
            The list of X, Y positions
        """          
           
        #read the image and track csv files
        imgData, trkData = self.__readCSV()
        #rename some image data columns
        imgData = imgData.rename(
            columns={
                "fileName": "img_name",
                "name": "img_name",
                "beamON": "laserON",
                "M_I_mean": "I_mean_movmean",
                "M_I_mean_crop": "I_mean_crop_movmean",
            }
        )
        #Prepare feature data and target labels
        data, target, tracks, time, XYPos = self.__data_preparation(imgData, trkData, drop_tracks=drop_tracks, speed_vector=speed_vector, acc_calculation=acc_calculation, 
                                                             acc_window=acc_window, acc_thresholdGradient=acc_thresholdGradient, acc_filterG1Zero=acc_filterG1Zero, 
                                                             start_trackNr=start_trackNr, augmentation=augmentation, powder_flux=powder_flux)
        return data, target, tracks, time, XYPos

    def __tracks_train_test_split(self, data, target, kind="random", given_train_tracks=None, given_test_tracks=None, test_size=0.25, keep_tracks=False):
        """Splits the data and targets per track# into train and test sets

        Parameters
        ----------
        data : dataframe
            Dataframe containing the feature data
        target : dataframe
            Dataframe containing the target labels
        kind : str, optional
            Defines the kind of train and test split (default is 'random'):
            - random: the track# to split into train and test will be chosen randomly
            - given: the track# to split into train and test will be given by the parameters given_train_tracks and given_test_tracks
        given_train_tracks : str, optional
            List of track# to be used as train set if kind='given' (default is None)
        given_test_tracks : str, optional
            List of track# to be used as test set if kind='given' (default is None)
        test_size : decimal, optional
            If kind='random' this parameter defines the ratio for splitting into train and test (default is 0.25 which means 25% will be used for test)
        keep_tracks : boolean, optional
            Defines if the column 'track#' should be kept or dropped (default is False)

        Returns
        -------
        dataframe
            containing the train features
        dataframe
            containing the train labels
        dataframe
            containing the test features
        dataframe
            containing the test labels  
        list
            List of train track#
        list
            List of test track#

        Raises
        ------
        ValueError
            Raised if wrong kind is given. Allowed values are 'random' and 'given'.
        ValueError
            Raised if kind='given' but no tracks are passed by given_train_tracks or given_test_tracks         
        """       

        #check which kind of split we need
        if kind == "random":
            if len(data['track#'].unique()) > 1 and test_size>0:
                train_tracks, test_tracks = train_test_split(data['track#'].unique(), test_size=test_size, random_state=seed)
            else:
                train_tracks, test_tracks = data['track#'].unique(), []
        elif kind == "given":
            if given_train_tracks is not None:
                train_tracks = given_train_tracks
            else:
                raise ValueError("Parameter given_train_tracks must contain track numbers if kind 'given' is chosen!")
            if given_test_tracks is not None:
                test_tracks = given_test_tracks
            else:
                raise ValueError("Parameter given_test_tracks must contain track numbers if kind 'given' is chosen!")
        else:
            raise ValueError("You need to give a valid argument for kind. '{kind}' was given but allowed values are 'random' and 'given'!")
        
        Xtrain = data.loc[data["track#"].isin(train_tracks)].copy()
        ytrain = target.loc[data["track#"].isin(train_tracks)].copy()
        Xtest = data.loc[data["track#"].isin(test_tracks)].copy()
        ytest = target.loc[data["track#"].isin(test_tracks)].copy()

        #drop columns that we don't need for training. 
        if keep_tracks:
            to_drop = ['G1']
        else:
            to_drop = ['G1', 'track#']
        Xtrain.drop(to_drop, axis=1,inplace=True)
        Xtest.drop(to_drop, axis=1,inplace=True)
        
        return Xtrain, ytrain, Xtest, ytest, train_tracks, test_tracks

    def __scale_the_data(self, Xtrain, ytrain, Xtest, ytest, kind="MinMax", scaler_x=None, scaler_y=None):
        """Scale the data of the train and test feature and label sets

        Parameters
        ----------
        Xtrain : dataframe
            Dataframe containing the train features
        ytrain : dataframe
            Dataframe containing the train labels
        Xtest : dataframe
            Dataframe containing the test features
        ytest : dataframe
            Dataframe containing the test labels
        kind : str, optional
            Determine the type of scaler to be used (default is 'std'):
            - 'std' : StandardScaler()
            - 'MinMax' : MinMaxScaler()
            - 'robust' : RobustScaler()
            - 'quantile' : QuantileTransformer()
        scaler_x : Scaler, optional
            The scaler used for the features X
        scaler_y : Scaler, optional
            The scaler used for the labels y

        Returns
        -------
        dataframe
            containing the scaled train features
        dataframe
            containing the scaled train labels
        dataframe
            containing the scaled test features
        dataframe
            containing the scaled test labels  
        scaler
            The scaler for the features X
        scaler
            The scaler for the labels y        

        Raises
        ------
        ValueError
            Raised if wrong kind is given. Allowed values are 'std', 'MinMax', 'robust' or 'quantile'
        """     

        #Determine the type of scaler
        fit = False
        if scaler_x == None and scaler_y == None:
            fit=True
            if kind=="std":
                scaler_x, scaler_y = StandardScaler(), StandardScaler()
            elif kind=="MinMax":
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            elif kind=="robust":
                scaler_x, scaler_y = RobustScaler(), RobustScaler()
            elif kind=="quantile":
                scaler_x, scaler_y = QuantileTransformer(), QuantileTransformer
            else:
                raise ValueError("Wrong kind is given with '{kind}'. Allowed values are 'std', 'MinMax', 'robust' or 'quantile'!")

        #Scale the train and test features and labels
        dropcols = []
        if "track#" in Xtrain.columns:
            dropcols.append("track#")
        if "augmentation" in Xtrain.columns:
            dropcols.append("augmentation")
        if len(dropcols) > 0:
            if fit:
                Xtrain[Xtrain.columns.drop(dropcols)] = scaler_x.fit_transform(Xtrain[Xtrain.columns.drop(dropcols)])
                ytrain[ytrain.columns] = scaler_y.fit_transform(ytrain[ytrain.columns])
            else:
                Xtrain[Xtrain.columns.drop(dropcols)] = scaler_x.transform(Xtrain[Xtrain.columns.drop(dropcols)])
                ytrain[ytrain.columns] = scaler_y.transform(ytrain[ytrain.columns])               
            if Xtest.shape[0] > 0:
                Xtest[Xtest.columns.drop(dropcols)] = scaler_x.transform(Xtest[Xtest.columns.drop(dropcols)])
                ytest[ytest.columns] = scaler_y.transform(ytest[ytest.columns])
        else:
            if fit:
                Xtrain[Xtrain.columns] = scaler_x.fit_transform(Xtrain[Xtrain.columns])
                ytrain[ytrain.columns] = scaler_y.fit_transform(ytrain[ytrain.columns])
            else:
                Xtrain[Xtrain.columns] = scaler_x.transform(Xtrain[Xtrain.columns])
                ytrain[ytrain.columns] = scaler_y.transform(ytrain[ytrain.columns])
            if Xtest.shape[0] > 0:
                Xtest[Xtest.columns] = scaler_x.transform(Xtest[Xtest.columns])
                ytest[ytest.columns] = scaler_y.transform(ytest[ytest.columns])
        
        return Xtrain, ytrain, Xtest, ytest, scaler_x, scaler_y

    def __getTrainAndTestTimeseries(self, batch_size=64, time_steps=20, kindOfScale='MinMax', kindOfSplit='random', given_train_tracks=None, given_test_tracks=None, 
                                    test_size=0.25, keep_tracks=False, drop_columns=None, drop_tracks=None, acc_calculation=True, acc_window=4, 
                                    acc_thresholdGradient=20, acc_filterG1Zero=True, speed_vector=False, start_trackNr=0, scaler_x=None, scaler_y=None, augmentation=False, powder_flux=0):
        """Returns the training and test feature and label timeseries

        Parameters
        ----------
        batch_size : int, optional
            Determines the batch size of the timeseries (default is 64)
        time_steps : int, optional
            Determines the window size of the timeseries (default is 20)
        kindOfScale : str, optional
            Determine the type of scaler to be used (default is 'std'):
            - 'std' : StandardScaler()
            - 'MinMax' : MinMaxScaler()
            - 'robust' : RobustScaler()
            - 'quantile' : QuantileTransformer()
        kindOfSplit : str, optional
            Defines the kind of train and test split (default is 'random'):
            - random: the track# to split into train and test will be chosen randomly
            - given: the track# to split into train and test will be given by the parameters given_train_tracks and given_test_tracks
        given_train_tracks : str, optional
            List of track# to be used as train set if kindOfSplit='given' (default is None)
        given_test_tracks : str, optional
            List of track# to be used as test set if kindOfSplit='given' (default is None)
        test_size : decimal, optional
            If kindOfSplit='random' this parameter defines the ratio for splitting into train and test (default is 0.25 which means 25% will be used for test)
        keep_tracks : boolean, optional
            Defines if the column 'track#' should be kept or dropped (default is False)
        drop_columns : list, optional
            Defines the column names of the data to be dropped (default is None)  
        drop_tracks : list, optional
            Defines the tracks not to be considered  
        acc_calculation: boolean, optional
            Defines if the calculation of acceleration/deceleration/stable have to take place               
        acc_window : int, optional
            The window size to be considered to determine gradient of velocity (default is 4)
        acc_thresholdGradient : int, optional
            Only if the velocity change within the defined window is above this threshold, it will be considered as acceleration or deceleration (default is 20)
        acc_filterG1Zero : boolean, optional
            Determines if the calculation have only to take place in case of G1=1 (default is True)    
        start_trackNr : int, optional
            Determines the number of the first track (default = 0)     
        scaler_x : Scaler, optional
            A Scaler to be used to scale the data
        scaler_y : Scaler, optional
            A Scaler to be used to scale the data     
        augmentation : boolean, optional
            Defines if a data aufmentation (rotation, mirroring) have to take place (default = False)  
        powder_flux : float, optional
            The amount of powder in g/s (default is 0)

        Returns
        -------
        timeseries
            containing the train features and labels as a time series with the given time window (time_steps)
        timeseries
            containing the test features and labels as a time series with the given time window (time_steps)
        scaler
            The scaler used for the features X
        scaler
            The scaler used for the labels y
        DataFrame
            The features dataframe
        DataFrame
            The labels
        list
            The list of tracks
        list
            The list of timestamps
        list
            The list of features
        list
            The list of X, Y positions
        list
            List of train track#
        list
            List of test track#
        list
            List of train trackidx
        list
            List of test trackidx
        """            

        #Prepare the data and target dataframes
        data, target, tracks, time, XYPos = self.__getDataAndTarget(drop_tracks=drop_tracks, acc_calculation=acc_calculation, acc_window=acc_window, 
                                                             acc_thresholdGradient=acc_thresholdGradient, acc_filterG1Zero=acc_filterG1Zero, 
                                                             speed_vector=speed_vector, start_trackNr=start_trackNr, augmentation=augmentation,
                                                             powder_flux=powder_flux)
        #Split into train and test sets
        Xtrain, ytrain, Xtest, ytest, train_tracks, test_tracks = self.__tracks_train_test_split(  data, 
                                                                        target, 
                                                                        kind=kindOfSplit, 
                                                                        given_train_tracks=given_train_tracks, 
                                                                        given_test_tracks=given_test_tracks, 
                                                                        test_size=test_size,
                                                                        keep_tracks=keep_tracks)
        #Drop some columns which are not needed anymore
        if drop_columns != None:
            for col in drop_columns:
                if col in Xtrain.columns:
                    Xtrain.drop(col, axis=1, inplace=True)
                    Xtest.drop(col, axis=1, inplace=True)                                                                       
        #Scale the data sets
        Xtrain, ytrain, Xtest, ytest, scaler_x, scaler_y = self.__scale_the_data(Xtrain, ytrain, Xtest, ytest, kind=kindOfScale, scaler_x=scaler_x, scaler_y=scaler_y)
        #Generate the time series for train and test
        ds_train, trackidx_train = DatasetGenerator(Xtrain, ytrain, time_steps=time_steps, batch_size=batch_size).getDataSet()
        ds_test, trackidx_test = DatasetGenerator(Xtest, ytest, time_steps=time_steps, batch_size=batch_size).getDataSet()
        #Feature and label names
        if augmentation:
            features = Xtrain.columns.drop(["track#", "augmentation"]).tolist()
        else:
            features = Xtrain.columns.drop("track#").tolist()
        labels = ytrain.columns.tolist()

        return ds_train, ds_test, scaler_x, scaler_y, data, target, tracks, time, features, labels, XYPos, train_tracks, test_tracks, trackidx_train, trackidx_test

class PrepareModel:
    """This class set up the model and trains it

    Attributes
    ----------
    ds_train : timeseries
        The timeseries containing the train features and labels
    ds_test : timeseries
        The timeseries containing the test features and labels
    nodes : int, optional
        Defines the number of LSTM nodes (default is 75)
    epochs : int, optional
        Defines the number of epochs to be trained (default is 30)
    LR : decimal, optional
        Defines the training learning rate (default is 0.0002)
    decay_steps : int, optional
        The maximum number of learning rate decay steps (default is 10000)
    decay_rate : decimal, optional
        The decay of the learning rate (default is 0.8 which means the new learning rate will be 80% of the previous learning rate)
    clipvalue : decimal, optional
        The gradient of each weight is clipped to be no higher than this value (default is 0.95)
    loss : str, optional
        The loss function used for training (default is 'mse')
    metrics : str, optional
        The metric used for training (default is 'mse')
    shuffle : boolean, optional
        Defines if the training data have to be shuffled (default is False)
    verbose : int, optional
        Defines which details of the training progress have to be printed (default is 1) 
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch

    Methods
    -------
    getModel()
        Returns the trained LSTM model
    getHistory()
        Returns the history of the model training
    plotLossHistory()
        Plots the loss curve of training and validation set
    getTestPrediction()
        Returns the prediction for the test timeseries
    predict(timeseries):
        Returns the prediction for the given timeseries
    """

    def __init__(self, 
                 ds_train, 
                 ds_test,
                 nodes=128,
                 epochs = 30, 
                 LR=1e-3,
                 decay_steps=1000,
                 decay_rate=0.8,
                 clipvalue=0.95,
                 loss=["mse"],
                 metrics=["mse"], 
                 shuffle=False,
                 verbose=1
                ):

        self.input_size = ds_train.element_spec[0].shape[-1]
        self.output_size = ds_train.element_spec[1].shape[-1]
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.nodes = nodes
        self.epochs = epochs
        self.LR = LR
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.clipvalue = clipvalue
        self.loss = loss
        self.metrics = metrics
        self.shuffle = shuffle
        self.verbose = verbose

        self.model, self.history = self.__buildAndTrainModel()

    def getModel(self):
        """Returns the trained model

        Returns
        -------
        model
            Trained Keras LSTM model
        """     

        return self.model    

    def getHistory(self):
        """Returns the training history

        Returns
        -------
        history
            History of the LSTM model training
        """     

        return self.history 

    def plotLossHistory(self):
        """Plots the loss curve of training and validation set
        """        

        setPltParams()
        plt.plot(self.history.history['loss'], label="loss")
        plt.plot(self.history.history['val_loss'], label="val_loss")
        plt.legend()
        plt.show()

    def getTestPrediction(self):
        """Returns the prediction for the test timeseries

        Returns
        -------
        array
            Prediction for the test timeseries
        """       

        return self.predict(self.ds_test)

    def predict(self, ds):
        """Returns the prediction for the given timeseries

        Parameters
        ----------
        ds : timeseries
            Timeseries features to execute prediction for

        Returns
        -------
        array
            Prediction for the given timeseries
        """   

        return self.model.predict(ds)

    def __buildAndTrainModel(self):
        """Builds, compiles and train the LSTM model

        Returns
        -------
        model
            Keras LSTM model
        """        

        model = self.__defineModel()
        model = self.__compileModel(model)
        model, history = self.__fitModel(model)
        return model, history

    def __defineModel(self):
        """Builds the LSTM model

        Returns
        -------
        model
            Build Keras LSTM model
        """     

        inputs = tf.keras.layers.Input((None, self.input_size))
        lstm = tf.keras.layers.LSTM(self.nodes)(inputs)
        output = tf.keras.layers.Dense(self.output_size, activation='relu')(lstm)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model     

    def __compileModel(self, model):
        """Compiles the LSTM model

        Parameters
        -------
        model : model
            Keras LSTM model to be compiled

        Returns
        -------
        model
            Compiled Keras LSTM model
        """           

         # Compiling LSTM Network and preparing time series
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.LR,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate
                    )
                
        adam_with_lr_decay = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    clipvalue=self.clipvalue 
                    )

        model.compile(
            loss=self.loss, optimizer=adam_with_lr_decay, metrics=self.metrics
        )

        return model

    def __fitModel(self, model):
        """Train the LSTM model

        Parameters
        -------
        model : model
            Keras LSTM model to be trained

        Returns
        -------
        model
            Trained Keras LSTM model
        history
            The training history
        """           
        
        # Train LSTM
        history = model.fit(
            self.ds_train,
            epochs=self.epochs,
            shuffle=self.shuffle,
            validation_data=self.ds_test,
            verbose=self.verbose
        )

        return model, history   

class Analysis:
    """This class allows to execute some analysis based on the prepared data and model

    Attributes
    ----------
    preparedModel : PreparedModel
        The trained model
    scaler : Scaler
        The Scaler used to prepare the labels y
    labels : list
        The names of the labels
    ds : Dataset, optional
        Timeseries for which a model prediction will be executed (default is None which means the test data are used)
   featuretxt : list, optional
        The list of feature labels
    unitsf : list, optional
        The list of feature units
    unitsl : list, optional
        The list of label units

    Methods
    -------
    getGroundTruth():
        Returns the scaled or unscaled ground truth
    getPrediction():
        Returns the scaled or unscaled prediction        
    calculateMSE():
        Calculates the MSE of prediction and ground truth
    visualizePrediction()
        Visualize the prediction and the ground truth for a given timeseries
    scatterPrediction():
        Plots the prediction and the ground truth for the given timeseries in a scatter plot
    getCV():
        Runs several trainings to calculate the cross-validation 
    plotCV():
        Plots violines for the given CV MSEs
    plotAccelerationDecelerationStable():
        Plots the acceleration, deceleration and stable on two plots        
    plotPredictionsVsGroundTruth():
        Plots the prediction vs. the ground truth curve
    showFeatureImportance():
        Determine and plot the importance of each feature
    compareFeaturesAndTargets():
        Determine and plot the importance of each feature
    visualizeTrackError():
        Print the track path and visualize the error by color intensity
    """    

    def __init__(self, preparedModel, scaler, labels=["thickness", "height", "area"], ds=None,
                       featuretxt=["$I_{mean}$", "powder flux", "V", "P", "W-neighbors", "W/2-neighbors", "$V_{x}$", "$V_{y}$"], 
                       unitsf=["", "g/s", "m/s", "W", "count", "count", "m/s", "m/s"],
                       unitsl=["mm", "mm", "mm"]):
        self.preparedModel = preparedModel
        self.labels = labels
        self.scaler = scaler
        self.featuretxt = featuretxt
        self.unitsf = unitsf
        self.unitsl = unitsl
        if ds != None:
            self.ds = ds
        else:
            self.ds = preparedModel.ds_test

        #Execute prediction and collect ground truth from timeseries  
        self.y_test_ts, self.y_pred_LSTM = self.__collectPredictionAndGroundtruth(self.ds, self.preparedModel)

        #Inverse the scaling
        self.y_test_unscaled = self.scaler.inverse_transform(self.y_test_ts)
        self.y_pred_unscaled_LSTM = self.scaler.inverse_transform(self.y_pred_LSTM)

        #Sets the plot parameters
        setPltParams()

    def getGroundTruth(self, scaled=True):
        """Returns the scaled or unscaled ground truth

        Parameters
        ----------
        scaled : boolean, optional
            Calculate on scaled data (default is True)
        """    

        if scaled:
            return self.y_test_ts
        else:
            return self.y_test_unscaled

    def getPrediction(self, scaled=True):
        """Returns the scaled or unscaled prediction 

        Parameters
        ----------
        scaled : boolean, optional
            Calculate on scaled data (default is True)
        """    

        if scaled:
            return self.y_pred_LSTM
        else:
            return self.y_pred_unscaled_LSTM            

    def calculateError(self, labelNr=None, printout=True, scaled=True, mode="mse"):
        """Calculates the MSE of prediction and ground truth

        Parameters
        ----------
        labelNr : int, optional
            Allows to filter for specific label columns (default is None which means the MSE is calculated for all the labels)
        printout : boolean, optional
            Defines if the MSE have to be printed (default is True)
        scaled : boolean, optional
            Calculate on scaled data (default is True)
        mode : string, optional
            The type of error to be calculated [mse, mae] (default is mse)
        """    

        #Calculate the MSE for the scaled or unscaled prediction and ground truth 
        if scaled:
            y_test = self.y_test_ts
            y_pred = self.y_pred_LSTM
        else:
            y_test = self.y_test_unscaled
            y_pred = self.y_pred_unscaled_LSTM

        if labelNr is not None:
            y_test = y_test[:, labelNr]
            y_pred = y_pred[:, labelNr]

        if mode == "mse":
            e = mean_squared_error(y_test, y_pred)    
        elif mode == "mae":
            e = mean_absolute_error(y_test, y_pred)    
        else:
            raise ValueError("Only mode mse or mae allowed!")
        
        if printout:
            print(f"{mode} (scaled={scaled}): {e}")

        return e

    def visualizePrediction(self, labelNr, title, xlim_from=None, xlim_to=None, ylim_from=None, ylim_to=None, color=None, label="LSTM", show=True, gt=True):
        """Plots the prediction and the ground truth for the given timeseries

        Parameters
        ----------
        labelNr : int
            Number of the label column
        title : string
            The title of the plot
        xlim_from : int, optional
            Defines the positive or negative start of the x-axis (default is None)
        xlim_to : int, optional
            Defines the end of the x-axis (default is None)
        ylim_from : int, optional
            Defines the positive or negative start of the y-axis (default is None)
        ylim_to : int, optional
            Defines the end of the y-axis (default is None)         
        color : color, optional
            The color used for the prediction plot
        label : string, optional
            The label of the prediction plot
        show : boolean, optional
            Defines if the plot have to be shown directly
        gt : boolean, optional
            Defines, if the ground truth have to be plotted as well   
        """     

        #Sets the plot parameters
        setPltParams()
        
        #Plot the prediction and ground truth curves
        if show:
            plt.figure(figsize=(15,5))
        plt.title(title)
        plt.plot(self.y_pred_unscaled_LSTM[:,labelNr], label=label, linewidth=2, color=color)
        if gt:
            plt.plot(self.y_test_unscaled[:,labelNr], label="Ground truth", linewidth=2, color="darkorange")
        if ylim_from is not None and ylim_to is not None:
            plt.ylim(ylim_from, ylim_to)        
        if xlim_from is not None and xlim_to is not None:
            plt.xlim(xlim_from, xlim_to)
        plt.xlabel("time steps")
        plt.ylabel(self.labels[labelNr] + " (" + self.unitsl[labelNr] + ")")
        plt.legend()
        if show:
            plt.show()

    def scatterPrediction(self, labelNr, lim_from=None, lim_to=None):
        """Plots the prediction and the ground truth for the given timeseries in a scatter plot

        Parameters
        ----------
        labelNr : int
            Number of the label column
        lim_from : int, optional
            Defines the positive or negative start of the x-axis and y-axis (default is None)
        lim_to : int, optional
            Defines the end of the x-axis and y-axis (default is None)        
        """    

        #Sets the plot parameters
        setPltParams()
        
        #Plot the prediction and ground truth as scatter plot
        plt.scatter(self.y_test_unscaled[:,labelNr], self.y_pred_unscaled_LSTM[:,labelNr], alpha=0.7)
        plt.title(self.labels[labelNr])
        plt.plot(
                np.linspace(-0.2, 2, num=10),
                np.linspace(-0.2, 2, num=10),
                c="tab:red",
                marker="",
                linewidth=2.5,
                linestyle="dashed",
        )
        plt.xlim(lim_from, lim_to)
        plt.ylim(lim_from, lim_to)
        plt.show()        

    def getCV(self, tracks, cv_columns, drop_columns=None, n_splits = 4, n_repeats = 5, random_state = 100):
        """Runs several trainings to calculate the CV 

        Parameters
        ----------
        tracks : int
            List of the track#
        cv_columns : string 
            List of columns to be looped for calculation of the CV
        drop_columns : string, otional
            List of columns to be dropped from dataset (default is None)
        n_splits : int, optional
            Defines the number of KFold splits (default is 4)
        n_repeats : int, optional
            Defines the number of KFold repeats (default is 5)      
        random_state : int, optional
            Defines the random state of KFold (default is 100)      
        """                           
        
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        mse = []
        features = []
        cvtraintracks = []
        cvtesttracks = []

        with warnings.catch_warnings(record=True):
            # repeat for each column to drop
            for col in cv_columns:     
                columns = cv_columns.copy()
                columns.remove(col)
                if drop_columns != None:
                    columns += drop_columns
                print("Col: {0} Drop: {1} CV: {2}".format(col, columns, cv_columns))

                # cross validate it
                for train, test in rkf.split(tracks):            
                    # keep track of the relative feature
                    features.append(col)    
                    #Prepare the data
                    preparedData = PrepareData(kindOfSplit='given', given_train_tracks=train, given_test_tracks=test, keep_tracks=True, kindOfScale='MinMax', drop_columns=columns)
                    #Prepare the model
                    preparedModel = PrepareModel(preparedData.getDSTrain(), preparedData.getDSTest(), verbose=0)
                    #Collect prediction and ground truth
                    y_test_ts, ypred = self.__collectPredictionAndGroundtruth(preparedData.getDSTest(), preparedModel)
                    # append MSE
                    mse.append(mean_squared_error(y_test_ts, ypred))   
                    #collect tracks
                    cvtraintracks.append(train)
                    cvtesttracks.append(test)
        # Put results in a DataFrame
        CV = pd.DataFrame(data={"mse": mse, "Feature": features, "TrainTracks": cvtraintracks, "TestTracks": cvtesttracks})       
        return CV        

    def plotCV(self, MSEs, labels=["LSTM"], colors=["tab:blue"], saveTo=None):
        """Plots violines for the given CV MSEs 

        Parameters
        ----------
        MSEs : array
            List of the MSE to be plotted
        labels : string, optional
            List of the labels for the passed MSEs (default is LSTM)
        colors : string, optional
            List of the colors for the passed MSEs (default is tab:blue)      
        saveTo : string, optional
            The path to save the plot (default is None)      
        """   

        #Sets the plot parameters
        setPltParams()

        fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
        violins = ax.violinplot(MSEs, vert=False, showmedians=True)
        if len(MSEs) == len(colors):
            for i in range(len(MSEs)):
                violins["bodies"][i].set_facecolor(colors[i])
                violins["bodies"][i].set_alpha(0.7)
        for partname in ('cmaxes', 'cmins', 'cbars', 'cmedians'):
            violins[partname].set_edgecolor('black')
            violins[partname].set_linewidth(1.2)
        ax.set_yticks(range(1, len(MSEs)+1))
        ax.set_yticklabels(labels, fontdict={"multialignment": "center"})
        ax.set_title("Cross-Validation")
        ax.set_xlabel("MSE")
        plt.grid(b=False, axis="y")
        plt.grid(b=True, axis="x")
        if saveTo != None:
            fig.savefig(saveTo, format='eps', bbox_inches='tight')
        plt.show()

    def plotAccelerationDecelerationStable(self, data, ylim, xlim_left, xlim_right=None, linewidth = 3, saveTo=None):
        """Plots the acceleration, deceleration and stable on two plots

        Parameters
        ----------
        data : Dataframe
            Track data
        xlim_left : (int, int)
            Time index window to be plotted on left subplot (from, to) 
        xlim_right : (int, int)
            Time index window to be plotted on right subplot (from, to) 
        ylim : (int, int)
            Limits to be shown on y-axis for the right subplot (from, to) 
        linewidth : int, optional
            Width of the ploted curves (default is 3)
        saveTo : string, optional
            The path to save the plot (default is None)              
        """  

        #Sets the plot parameters
        setPltParams()

        #Create a plot consisting of subplots
        if xlim_right==None:
            numplots = 1
        else:
            numplots = 2
        fig, ax = plt.subplots(1, numplots, sharey=True, figsize=(11, 5))
        fig.subplots_adjust(wspace=0.02)

        # plot the same data on both axes and set right xlim and ylim
        if numplots==1:
            ax = [ax]
            self.__plotAccelerationDecelerationStableSubplot(data, ax[0], xlim_left, ylim, linewidth, True, True)
        else:
            self.__plotAccelerationDecelerationStableSubplot(data, ax[0], xlim_left, ylim, linewidth, True, False)
            self.__plotAccelerationDecelerationStableSubplot(data, ax[1], xlim_right, ylim, linewidth, False, True)

        # set labels
        ax[0].set_ylabel("Accelerating/decelerating/stable")
        ax[numplots-1].right_ax.set_ylabel("V (m/s)", rotation=270, labelpad=20)
        ax[0].set_xlabel("t (ms)", position=(1, 1))

        # hide the spines between ax[0] and ax[1]
        if numplots != 1:
            ax[0].spines["right"].set_visible(False)
            ax[0].right_ax.spines["right"].set_visible(False)
            ax[1].spines["left"].set_visible(False)
            ax[1].right_ax.spines["left"].set_visible(False)
            ax[0].tick_params(right=False, labelright=False)
            ax[0].right_ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
            ax[1].tick_params(right=False, left=False, labelright=False, labelleft=False)
            ax[1].right_ax.tick_params(left=False, right=True, labelleft=False, labelright=True)

        # make the legend taking the right lines out of the plot
        if numplots == 1:
            lines_0 = ax[0].get_lines() + ax[0].right_ax.get_lines()
            lines = [lines_0[0], lines_0[1], lines_0[2], lines_0[3]]
        else:
            lines_0 = ax[0].get_lines() + ax[0].right_ax.get_lines()
            lines_1 = ax[1].get_lines() + ax[1].right_ax.get_lines()
            lines = [lines_0[0], lines_1[0], lines_0[1], lines_0[2]]
        ax[numplots-1].legend(lines, ["accelerating", "decelerating", "stable", "speed"], loc="lower left")

        if numplots != 1:
            # Makes the cut-out diagonal lines look better and makes the automatically rescale with the plot (from https://matplotlib.org/3.3.2/gallery/subplots_axes_and_figures/broken_axis.html#sphx-glr-gallery-subplots-axes-and-figures-broken-axis-py)
            d = 0.015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax[0].transAxes, color="k", clip_on=False)
            ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
            ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax[1].plot((-d, +d), (-d, +d), **kwargs)

        if saveTo != None:
            fig.savefig(saveTo, format='eps', bbox_inches='tight')

        plt.show()

    def __plotAccelerationDecelerationStableSubplot(self, data, ax, xlim, ylim, linewidth, plot_acc, plot_dec):
        """Plots the acceleration, deceleration and stable on a subplot

        Parameters
        ----------
        data : Dataframe
            Track data
        ax : subplot
            Subplot used to plot
        xlim : (int, int)
            Time index window to be plotted (from, to)
        ylim : (int, int)
            Limits to be shown on y-axis
        linewidth : int
            Width of the ploted curves
        plot_acc : boolean
            Plot the acceleration
        plot_dec : boolean
            Plot the deceleration            
        """   
               
        if plot_acc:
            data["accelerating"].plot(ax=ax, xlim=xlim, color="tab:green", linewidth=linewidth)
        if plot_dec:
            data["decelerating"].plot(ax=ax, xlim=xlim, color="tab:orange", linewidth=linewidth)
        data["stable"].plot(ax=ax, xlim=xlim, color="tab:red", linewidth=linewidth)
        data["V"].plot(ax=ax, secondary_y=True, xlim=xlim, color="tab:blue", linewidth=linewidth, linestyle=(0, (5, 1)))
        ax.right_ax.set_ylim(ylim)

    def plotPredictionsVsGroundTruth(self, preparedData, selected_tracks, plot_NN_columns_pred = ["y_pred_D", "y_pred_H", "y_pred_A"], plot_NN_columns_test = ["y_test_D", "y_test_H", "y_test_A"], saveTo=None):
        """Plots the prediction vs. the ground truth curve

        Parameters
        ----------
        preparedData : PrepareData
            Track data object
        selected_tracks : list
            The list of track# to be displayed
        plot_NN_columns_pred : list, optional
            The list of prediction label columns           
        plot_NN_columns_test : list, optional
            The list of ground truth label columns       
        saveTo : string, optional
            The path to save the plot (default is None)                         
        """  

        #Sets the plot parameters
        setPltParams()

        time = preparedData.getTime(windowed=True).loc[preparedData.getData(windowed=True)["track#"].isin(selected_tracks)]
        y_pred_unscaled_NN = self.getPrediction(scaled=False)
        y_test_unscaled_NN = self.getGroundTruth(scaled=False)

        plot_NN_time = pd.DataFrame(
            data={
                "t": time.round(decimals=0).values,
                "y_pred_D": y_pred_unscaled_NN[:, 0],
                "y_pred_H": y_pred_unscaled_NN[:, 1],
                "y_pred_A": y_pred_unscaled_NN[:, 2],
                "y_test_D": y_test_unscaled_NN[:, 0],
                "y_test_H": y_test_unscaled_NN[:, 1],
                "y_test_A": y_test_unscaled_NN[:, 2],
                "track#": preparedData.getData(windowed=True)["track#"].loc[preparedData.getData(windowed=True)["track#"].isin(selected_tracks)].values
            }
        )

        # subplot letters size (a, b, c ...) and positions
        subplot_font_size = 10
        subplot_x = 0.49
        subplot_y = -0.18

        # subplots with axis breaks spacing
        subplot_top = 0.95
        subplot_wspace = 0.02
        subplot_hspace = 0.25

        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle("Prediction vs. ground truth")
        ylabel = ["W (mm)", "H (mm)", "A (mm$^2$)"]
        ylim = [(0, 1.4), (0, 1), (0, 0.35)]
        tracks = [6, 7, 8]

        # for over targets (D, H, A)
        for t in range(3):
            # for over the 3 tracks (they are consecutives, but this is not a requirement)
            for i in range(3):
                ax[t, i].plot(
                    plot_NN_time["t"].loc[plot_NN_time["track#"] == tracks[i]],
                    plot_NN_time[plot_NN_columns_test[t]].loc[
                        plot_NN_time["track#"] == tracks[i]
                    ],
                    label="Ground Truth",
                    color="tab:red",
                    linewidth=2.5,
                    linestyle=(0, (5, 1)),
                )
                ax[t, i].plot(
                    plot_NN_time["t"].loc[plot_NN_time["track#"] == tracks[i]],
                    plot_NN_time[plot_NN_columns_pred[t]].loc[
                        plot_NN_time["track#"] == tracks[i]
                    ],
                    label="Prediction",
                    color="tab:blue",
                    linewidth=2,
                )
                ax[t, i].set_ylim(ylim[t])
                if i == 0 or i == 1:
                    ax[t, i].spines["right"].set_visible(False)
                if i == 1 or i == 2:
                    ax[t, i].spines["left"].set_visible(False)
                    ax[t, i].tick_params(
                        right=False, labelright=False, left=False, labelleft=False
                    )

            d = 0.015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax[t, 0].transAxes, color="k", clip_on=False)
            ax[t, 0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax[t, 0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            kwargs.update(transform=ax[t, 1].transAxes)  # switch to the middle axes
            ax[t, 1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax[t, 1].plot((-d, +d), (-d, +d), **kwargs)
            ax[t, 1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax[t, 1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            kwargs.update(transform=ax[t, 2].transAxes)  # switch to the right axes
            ax[t, 2].plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax[t, 2].plot((-d, +d), (-d, +d), **kwargs)

            # set labels
            ax[t, 1].set_xlabel("t (ms)")
            ax[t, 0].set_ylabel(ylabel[t])

            # set letters to reference the figure (a, b, c)
            ax[t, 1].text(
                subplot_x,
                subplot_y,
                string.ascii_lowercase[t] + ")",
                transform=ax[t, 1].transAxes,
                size=subplot_font_size,
            )

            # add legend
            ax[t, 2].legend(loc="upper right")

        fig.tight_layout()
        fig.subplots_adjust(top=subplot_top, wspace=subplot_wspace, hspace=subplot_hspace)

        if saveTo != None:
            fig.savefig(saveTo, format='eps', bbox_inches='tight')

        plt.show()

    def showFeatureImportance(self, preparedModel, ds, features, title, color=None):
        """Determine the permutation feature importance

        Parameters
        ----------
        preparedModel : PreparedModel
            The trained model used for the prediction
        ds : Dataset
            Timeseries for which a model prediction will be executed (have to be passed as numpy array)
        features : list
            The list of features for which the importance have to be determined
        title : string
            The title of the plot
        color : color
            The color of the plot
        """   

        #Sets the plot parameters
        setPltParams()         
        
        #Collect X and y
        X = np.concatenate(ds[:, 0])
        y = np.concatenate(ds[:, 1])
        results = []
        #Determine the mse for baseline
        baseline_mse = mean_squared_error(y, preparedModel.predict(X))
        results.append({'feature':'BASELINE','mse':baseline_mse})           
        #Determine the mse for each shuffled feature
        l = len(features)
        for k in range(l):
            X_ = X.copy()
            np.random.shuffle(X_[:,:,k])
            mse = mean_squared_error(y, preparedModel.predict(X_))
            results.append({'feature':self.featuretxt[k],'mse':mse})
        #Plot the mse per feature
        df = pd.DataFrame(results).sort_values('mse')
        plt.figure(figsize=(10,5))
        plt.barh(np.arange(l+1),df.mse, color=color)
        plt.yticks(np.arange(l+1),df.feature.values)
        plt.title(f'Model feature importance ({title})')
        plt.ylim((-1,l+1))
        plt.plot([baseline_mse,baseline_mse],[-1,l+1], '--', color='orange', label=f'Baseline \nMSE={baseline_mse:.3f}')
        plt.ylabel('Feature')
        plt.xlabel('MSE')
        plt.legend()
        plt.grid(b=False, axis="y")
        plt.show()

    def compareFeaturesAndTargets(self, Xtrain, ytrain, Xpred, ypred, features, labels):
        """Determine and plot the importance of each feature

        Parameters
        ----------
        Xtrain : DataFrame
            The train data
        ytrain : DataFrame
            The train labels
        Xtest : DataFrame
            The test data
        ytest : DataFrame
            The test labels
        features: list
            The list of features
        labels : list
            The list of labels
        """   

        #Sets the plot parameters
        setPltParams()

        for i in range(len(features)):
            feature = features[i]
            _, ax = plt.subplots(1, 1, figsize=(11, 3.5))
            ax.violinplot([Xpred[feature], Xtrain[feature]], vert=False, showmedians=True)
            plt.title("Feature: {0}".format(self.featuretxt[i]))
            ax.set_yticks(range(1, 3))
            ax.set_yticklabels(["Test", "Train"], fontdict={"multialignment": "center"})
            ax.set_xlabel(self.unitsf[i])
            plt.grid(b=False, axis="y")
            plt.grid(b=True, axis="x")
            plt.show()
        for i in range(len(labels)):
            label = labels[i]
            _, ax = plt.subplots(1, 1, figsize=(11, 3.5))
            violin_parts = ax.violinplot([ypred[label], ytrain[label]], vert=False, showmedians=True)
            for vp in violin_parts['bodies']:
                vp.set_facecolor('#bd9dc4')
            plt.title("Target: {0}".format(self.labels[i]))
            ax.set_yticks(range(1, 3))
            ax.set_yticklabels(["Test", "Train"], fontdict={"multialignment": "center"})
            ax.set_xlabel(self.unitsl[i])
            plt.grid(b=False, axis="y")
            plt.grid(b=True, axis="x")
            plt.show()

    def plotHvsVP(self, preparedData):
        """Print the H of single tracks in dependence of V and Pnom

        Parameters
        ----------
        preparedData : preparedData
            The preparedData object
        """

        #Sets the plot parameters
        setPltParams()

        singletrackdata = preparedData.getData()
        singletracktarget = preparedData.getTarget()
        singletrack = pd.concat([singletrackdata, singletracktarget], axis=1)

        Vs = np.asarray([300, 450, 600, 750, 900, 1050]) / 60000
        Ps = np.asarray([200, 300, 400, 500, 600, 700])
        for p in range(len(Ps)):
            pt = singletrack[singletrack.Pnom.between(Ps[p]-50, Ps[p]+50)]
            m = []
            for v in range(len(Vs)):
                vt = pt[pt.V.between(Vs[v]-75/60000, Vs[v]+75/60000)]
                m.append(vt.H.mean()) 
            plt.plot(Vs, m, label=str(Ps[p])+" W", marker='o')
        plt.ylabel('Height (mm)')
        plt.xlabel('V (m/s)')
        plt.xticks(Vs)
        plt.legend()
        plt.show()

    def visualizeTrack(self, preparedData, labelNr, trackNrs, title="Random track #", inOnePlot=True, mode="error", excerpts=[]):
        """Print the track path and visualize the error, ground truth or prediction by color intensity

        Parameters
        ----------
        preparedData : preparedData
            The preparedData object
        labelNr : int
            The label number
        trackNrs : list
            The list of track#
        title : string, optional
            The title to be added to the plot
        inOnePlot : boolean, optional
            Defines of all tracks have to be plotted in one or in single plots (default is True)
        mode : string, optional 
            Defines the mode to be displayed ["error", "ground truth", "prediction"] (default is error)
        excerpts : list, optional
            The excerpts to be desplayed in the plot [[lim_from, lim_to], ...] (default is no excerpt)
        """       

        #Sets the plot parameters
        setPltParams()

        if preparedData.trackidx_test == []:
            idx = preparedData.trackidx_train
            scope="train"
        else:
            idx = preparedData.trackidx_test
            scope="test"

        label = self.labels[labelNr]

        if self.y_pred_unscaled_LSTM.shape[0] != idx.shape[0]:
            raise ValueError("The analysis prediction data don't match the track index!")
        
        for trackNr in trackNrs:
            if trackNr not in preparedData.getTracks(scope=scope):
                raise ValueError("Track# not in prediction set!")

        pos_tracks = None
        for trackNr in trackNrs:
            pos_track = preparedData.getXYPos()[preparedData.getData()["track#"]==trackNr][preparedData.time_steps-1:].copy()
            if mode == "error":
                d = np.abs(self.y_pred_unscaled_LSTM[idx == trackNr,labelNr] - self.y_test_unscaled[idx == trackNr,labelNr])
            elif mode == "ground truth":
                d = self.y_test_unscaled[idx == trackNr,labelNr]
            elif mode == "prediction":
                d = self.y_pred_unscaled_LSTM[idx == trackNr,labelNr]
            else:
                raise ValueError("Only one of the following modes are allowed: [error, ground truth, prediction]!")
            pos_track[mode] = d
            if inOnePlot:
                if pos_tracks is None:
                    pos_tracks = pos_track
                else:
                    pos_tracks = pd.concat([pos_tracks, pos_track], axis=0)
            else:
                self.__scatterTrackError(pos_track, label, trackNr, title, mode, excerpts)
        if inOnePlot:
            self.__scatterTrackError(pos_tracks, label, trackNrs, title, mode, excerpts)
        
        sns.set(font_scale=1)

    def __scatterTrackError(self, pos_track, label, trackNr, title, mode, excerpts=[]):
        """Scatter the track error

        Parameters
        ----------
        pos_track : Dataframe
            Position data and error
        label : string
            The label text
        trackNr : int
            The number of the track
        title : string
            The title to be added to the plot
        mode : string 
            Defines the mode to be displayed ["error", "ground truth", "prediction"]
        excerpts : list, optional
            The excerpts to be desplayed in the plot [[x, y, w, h], ...] (default is no excerpt)    
        """      

        #Sets the plot parameters
        setPltParams()

        _, axes = plt.subplots(1,1, figsize=(15, 6))
        sns.scatterplot(ax=axes, 
                        data=pos_track, 
                        x="Xpos" , y="Ypos",
                        hue=mode,
                        hue_norm=(0,0.75),
                        palette="flare",
                        linewidth=0,
                        alpha = 0.3,
                        legend="brief")
        axes.set(xlabel='Xpos (mm)', ylabel="Ypos (mm)")
        axes.set_aspect('equal')
        axes.set_title(f"{title}{trackNr}: {label} {mode}")
        axes.set_xlim(pos_track["Xpos"].min()-1, pos_track["Xpos"].max()+1)
        axes.set_ylim(pos_track["Ypos"].min()-1, pos_track["Ypos"].max()+1)
        axes.grid(False)
        if mode=="error":
            label = mode
        plt.legend(title=label+" values", bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        #Print the excerpt rectangles
        for excerpt in excerpts:
            lim_from, lim_to = excerpt
            x = pos_track.iloc[lim_from:lim_to].Xpos.min()-.3
            y = pos_track.iloc[lim_from:lim_to].Ypos.min()-.3
            w = pos_track.iloc[lim_from:lim_to].Xpos.max() - x + .6
            h = pos_track.iloc[lim_from:lim_to].Ypos.max() - y + .6
            rect = Rectangle((x, y), w, h, facecolor='none', edgecolor='darkgrey', linewidth=2, linestyle = 'dashed')
            axes.add_patch(rect)
        plt.show()

    def __collectPredictionAndGroundtruth(self, ds, preparedModel):
        """Collect prediction and ground truth 

        Parameters
        ----------
        ds : Dataset
            Timeseries for which a model prediction will be executed
        preparedModel : PreparedModel
            The trained model
        """           

        #Execute prediction and collect ground truth from timeseries  
        y_test_ts=np.empty((0, len(self.labels)))
        y_pred_LSTM = preparedModel.predict(ds)
        for _, y in ds:
            y_test_ts=np.append(y_test_ts, y, axis=0)
        return y_test_ts, y_pred_LSTM

