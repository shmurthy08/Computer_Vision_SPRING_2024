import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
import time
# Define directory containing the dataset
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CamSeq07")

# Load preprocessed data
preprocessed_data = np.load("preprocessed_data_for_RF.npz")
X_train, y_train = preprocessed_data['X_train'], preprocessed_data['y_train']



# Start the timer
start_time = time.time()

rf = RandomForestClassifier(n_estimators=3, random_state=42)
rf.fit(X_train, y_train)

# Start the timer
end_time = time.time()
# Calculate the runtime
runtime = end_time - start_time
print("Training runtime: {:.2f} seconds".format(runtime))

# Save the model
with open("rf_model.pkl", "wb") as f:
    pkl.dump(rf, f)
    
