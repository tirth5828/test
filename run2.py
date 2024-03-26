import os
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , f1_score
from tqdm import tqdm
import json

import warnings
warnings.simplefilter(action='ignore')



_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def maha(M, phi, ai):
    return np.dot(np.dot(ai - phi, np.linalg.inv(M)), ai - phi)

def hellinger1(M , p, q):
      return norm(np.sqrt(np.abs(p)) - np.sqrt(np.abs(q))) / _SQRT2



def hellinger2(M , p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(M , p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

distances = [maha , hellinger1 , hellinger2 , hellinger3 ]


def parametric_filter(streaming_points, r, f , regularization=1e-6):
    n = len(streaming_points)
    C = []
    Omega = []
    index = []
    M = np.zeros((len(streaming_points[0]), len(streaming_points[0])))
    phi = np.zeros(len(streaming_points[0]))
    S = 0
    lambda_val = min(streaming_points[0])
    nu = max(streaming_points[0])

    for i in range(1,n+1):
        lambda_val = min(lambda_val,min(streaming_points[i-1]))
        nu = max(nu, max(streaming_points[i-1]))

        M = np.linalg.inv((1 / nu) * M + np.outer(streaming_points[i-1], streaming_points[i-1]) / n + regularization * np.eye(len(M)))
        mu = lambda_val / nu
        phi = ((i - 1) * phi + streaming_points[i-1]) / (i )
        S += f(M,phi, streaming_points[i-1])

        if i == 0:
            p = 1
        else:
            l = ( ((2 * f(M, phi, streaming_points[i-1])) / (mu * S)) + (8 / (mu * (i -1))))
            p = min(1, r * l)

        if np.random.rand() < p:
          C.append(streaming_points[i-1])
          index.append(i-1)
          if p != 0:
              omega = 1 / p
          else:
              omega = 0
          Omega.append(omega)

    return C, Omega, index


def load_data(root_dir):
    data = []
    labels = []

    # Iterate over each folder in the root directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Get the label from the folder name
            label = folder
            
            # Iterate over each file in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                
                # Ensure it's a file and has the correct extension
                if os.path.isfile(file_path) and file.endswith('.jpeg'):
                    # Load the image
                    img = Image.open(file_path)
                    
                    img.resize((10, 10))
                    # Convert the image to numpy array
                    img_array = np.array(img)
                    # Append the image array and its label to the data list
                    data.append(img_array)
                    labels.append(label)
    
    return np.array(data), np.array(labels)

# Example usage:
root_dir = 'archive'
data, labels = load_data(root_dir)
print("Data shape:", data.shape)
print("Labels:", labels)

def preprocess_data(data, labels, test_size=0.33):
    # Flatten the image data
    data_flattened = data.reshape(data.shape[0], -1)
    
    # Min-Max Normalization
    data_normalized = (data_flattened - np.min(data_flattened)) / (np.max(data_flattened) - np.min(data_flattened)) + 1
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(data, labels)


# Print the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



params = np.linspace(70,  0.001 , 10)


corset_length = []
accuracy_scores = []
data = []
for dist in tqdm(distances, desc="Distances"):  
    for r in tqdm(params, desc="Parameter r"):
        Coreset, Weights , index = parametric_filter(np.array(X_train), r , dist)
        corset_length.append(len(Coreset))
        
        
        Coreset_features = np.array(X_train)[index]
        Coreset_lables = np.array(y_train)[index]
        
        svc = SVC(kernel='rbf',  # Change kernel for multiclass
                 C=1.0, random_state=42)
        
        try:
            svc.fit(Coreset_features, Coreset_lables)

            y_pred = svc.predict(X_test)

            score =  f1_score(y_test, y_pred, average='weighted')

            accuracy_scores.append(score)
            
            data_point = {
                "dist_function": str(dist),  # Convert function to string for JSON
                "r": r,
                "index": index,  # Convert index array to list for JSON
                "accuracy_score": score
            }
            data.append(data_point)
        except:
            data_point = {
                "dist_function": str(dist),  # Convert function name to string
                "r": r,
                "index": None,  # Indicate unsuccessful filtering with None
                "accuracy_score": 0.0  # Set accuracy to 0 for failed attempts
            }
            data.append(data_point)
            accuracy_scores.append(0)
 
 
with open("Medical MNIST coreset_data.json", "w") as outfile:
    json.dump(data, outfile) 
  
  

# Plot Silhouette Score vs. Parameter for each distance metric
plt.figure(figsize=(10, 6))
for i, dist in enumerate(distances):
    plt.plot(params, accuracy_scores[i * len(params):(i + 1) * len(params) ], label=dist)
    print(f"Max score for {dist} is",max(accuracy_scores[i * len(params):(i + 1) * len(params) ]) )
plt.title('Accuracy Score vs. Parameter')
plt.xlabel('Parameter')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)
plt.savefig('Medical MNIST Accuracy Score vs. Parameter.png')

# Plot Corset Length vs. Parameter for each distance metric
plt.figure(figsize=(10, 6))
for i, dist in enumerate(distances):
    plt.plot(params, corset_length[i * len(params):(i + 1) * len(params) ], label=dist)
plt.title('Corset Length vs. Parameter')
plt.xlabel('Parameter')
plt.ylabel('Corset Length')
plt.legend()
plt.grid(True)
plt.savefig('Medical MNIST Corset Length vs. Parameter.png')

