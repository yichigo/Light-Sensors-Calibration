


# Run the Codes with the Preprocessed Data from Zenodo

## File Path:

place the codes in this Repository into ./src/ folder, for example:

	./src/model_MLP_whole.ipynb

place the preprocessed data files from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4946314.svg)](https://doi.org/10.5281/zenodo.4946314)
into ./data/ folder, for example:

	./data/10004098_001e06305a6b.csv

create this folder to save figures:

	./figures/

create this folder to save the trained models:

	./models/


## Run the ANN model

in jupyter notebook, run:

	model_MLP_whole.ipynb

if you are running the model for the default node '001e06305a6b' with data file '10004098_001e06305a6b.csv', you don't need to do any modification in the codes;

Otherwise, you might need to manually pick up appropriate datetimes in the testing dataset to "Compare Actual Spectrum and Estimated Spectrum" in a typical weather condition, like sunny or cloudy.





# Run the Codes from the Raw Data Files

If you are researchers in our MINTS team, and want to repeat all the work from the raw data file, please follow these steps:

## File Path:

place the codes in this Repository into ./src/ folder, for example:

	./src/model_MLP_whole.ipynb

Data of cheap light sensors:

	./lightsensors/node_id/year/month/day/...,  where node_id = '001e06305a6b' or others

Data of Minolta sensor:

	./Minolta/node_id/year/month/day/...,  where node_id = '10004098'

Data of GPS sensor:

	./Minolta/node_id/year/month/day/...,  where node_id = '001e0610c2e9'

create this folder to save the preprocessed data files:

	./data/

create this folder to save figures:

	./figures/

create this folder to save the trained models:

	./models/


## Data Preprocessing:

To prepare the data of Minolta sensor, run:

	python data_minolta.py

To visualize the daily spectrum, run:

	python daily_spectrum.py

To add solar angles (Zenith angle and Azimuth angle), run:

	data_minolta_solarAngle.py

To prepare the data of GPS with Minolta sensor, run:

	python data_gps.py

To prepare the data of 1 cheap light sensor, run:

	python data_lightsensors.py node_id

for example, with node_id = '001e06305a6b';


To prepare the data of all the cheap light sensors, run:

	python runall_data_lightsensors.py


After running the above steps, we can merge the data of Minolta sensor, GPS sensor, and other cheap light sensors by running:

	python data_merge.py



## Run the model

Here we recommand the multilayer perceptron (MLP) model, since it is small and fast.

For single wavelength model, run:

	model_MLP_single.ipynb

For whole spectrum model, run:

	model_MLP_whole.ipynb


Random Forest and XGB are also good for single wavelength prediction. However, for the whole spectrum prediction, the size of the models are too large and need a long time for tranning.


## Others

To check the GPS location, run:

	cheapSensors.ipynb

An example of cheap sensor visaulization, run:

	cheapSensors.ipynb




[![DOI](https://zenodo.org/badge/244495599.svg)](https://zenodo.org/badge/latestdoi/244495599)

