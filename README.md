# Frontiers Open Code LMD

## Recurrent neural networks for modeling two-dimensional track geometry in laser metal deposition

## Description
This repository contains the code used to produce the results published in the paper "Recurrent neural networks for modeling track geometry in laser metal deposition" (Martina Perani, Ralf Jandl, Stefano Baraldo, Anna Valente and Beatrice
Paoli), submitted in Frontiers in Artificial Intelligence. The raw data are published on Zenodo with an open access licence and are not provided with this repository.

## Usage
To reproduce the results published in the paper, the test_notebook.ipynb can be executed. This notebook is structured as:
1) Import of utils and parameter definition
   (all the following functionalities are defined in utilities_frontiers_AI.py and are used from the test_notebook.ipynb.)
2) Experiment 1 - training only with simple geometries (single tracks, V-tracks, spiral tracks)
   - Preparation of data and training of model
   - Analysis with test dataset
   - Analyses with validation dataset
   - Comparision of the datasets
3) Experiment 2 - training additionaly with one random track
   - Preparation of data and training of model
   - Analysis with test dataset
   - Analyses with validation dataset
   - Comparision of the datasets
4) Comparision of both experiments
5) Performance test

As mentioned, the data used are not provided with this repository but can be downloaded at the following links:
- single tracks: https://zenodo.org/record/3978982#.Y9gdvbLMIV8
- V-tracks: https://zenodo.org/record/3980733#.Y9gd07LMIV8
- Spiral tracks: https://zenodo.org/record/4061502 and https://zenodo.org/record/4049842
- Random tracks: https://zenodo.org/record/5607279
From this links the image csv files and the track csv files have to be downloaded and stored in a subfolder "Data".

At the very beginning of the test_notebook.ipynb the list of files to be used in the experiments have to be defined with the following structure: (folder / imgfile / trackfile / list of tracks to be dropped / powder flux)

## Get in touch
If you are having trouble with the code or have built upon it and want to share your results with us, please feel free to contact the authors.

Martina Perani, Fernfachhoschschule Schweiz, [martina.perani@ffhs.ch](martina.perani@ffhs.ch)

Ralf Jandl, Fernfachhochschule Schweiz, [ralf.jandl@ffhs.ch](ralf.jandl@ffhs.ch)
