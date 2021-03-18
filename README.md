

## File Path:

Data of cheap light sensors:	./lightsensors/node_id/...

Data of Minolta sensor:			./Minolta/node_id/..., where node_id = '10004098'

Data of GPS sensor:				./Minolta/node_id/..., where node_id = '001e0610c2e9'

Merged data of Minolta sensor and cheap light sensors:

								./data/

Codes:				./src/

Figures:			./figures/

Trained Models:		./models/


## Data Preprocessing:

To prepare the data of Minolta sensor,

	run: python data_minolta.py

To prepare the data of GPS with Minolta sensor,

	run: python data_gps.py

To prepare the data of 1 cheap light sensor,

	run: python data_lightsensors.py node_id

To prepare the data of all the cheap light sensors,

	run: python runall_data_lightsensors.py

After running all above, we can merge the data of Minolta sensor and other cheap light sensors,

	run: python data_merge.py



## Run the model

Here we recommand the multilayer perceptron (MLP) model, since it is small and fast.

For single wavelength model:

	run model_MLP_single.ipynb

For whole spectrum model:

	run model_MLP_whole.ipynb


Random Forest and XGB are also good for single wavelength prediction. However, for the whole spectrum prediction, the size of the models are too large and need a long time for tranning.


## Others

To check the GPS location,

run: cheapSensors.ipynb

An example of cheap sensor visaulization,

run: cheapSensors.ipynb
