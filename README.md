# CleanSpot — Python 3.14 version

This version does NOT use TensorFlow.

## Technologies
- Python 3.14
- Flask
- MySQL
- Bootstrap
- OpenCV
- scikit-learn
- NumPy
- Pillow

## Install

From the CleanSpot folder:

    py -m pip install -r requirements.txt

If pip is missing:

    py -m ensurepip --upgrade

## Database

1. Start WAMP/MySQL.
2. Open phpMyAdmin.
3. Import database.sql.
4. The default app settings expect:
   host = localhost
   user = root
   password = empty
   database = cleanspot

Change app.py if your MySQL credentials are different.

## Train the ML model

Put images in:

    dataset/biodegradable/
    dataset/recyclable/
    dataset/residual/
    dataset/special/

Then run:

    py train_model.py

The model will be saved as:

    model/cleanspot_model.pkl

## Run

    py app.py

Open the local Flask address shown in the terminal.

## Important ML limitation

This is a school-project prototype using OpenCV feature extraction + Random Forest. It is not a deep-learning image model and may make incorrect predictions. Use a separate test set and report the actual accuracy produced by your training run.
