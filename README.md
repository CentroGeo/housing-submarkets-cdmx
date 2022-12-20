# housing-submarkets-cdmx
Supplementary material for peer review


This repository contains the scripts we used to perform the analysis in _An interdisciplinary framework for the comparison of geography aware housing market segmentation models using web scraped listing price data for Mexico City_

All scripts are in _src_ folder

* Accesibility.py Contains the code used to model accessibility to ammenities
* Clustering.py Contains all clustering code to define submarkets
* Outliers_LOF.py contains the code to filter outliers
* Prediction.py Contains the XGBoost model  fitting procedure
* Welch_Test.py Contains the code to obtain confidence intervals for prices in submarkets

All data needed to run the scripts can be found in the `data` folder.  