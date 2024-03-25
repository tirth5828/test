import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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



mushroom_data = pd.read_csv("mushrooms.csv")

label_encoder = LabelEncoder()

# Iterate over each column in the DataFrame
for col in mushroom_data.columns:
    # Check if the column dtype is object (i.e., categorical)
    if mushroom_data[col].dtype == 'object':
        # Fit label encoder and transform values
        mushroom_data[col] = label_encoder.fit_transform(mushroom_data[col]) + 1
        
        
        
# Extracting features and labels
mushroom_data_features = mushroom_data.drop(['class'] , axis = 1)
mushroom_data_labels = mushroom_data['class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mushroom_data_features, mushroom_data_labels, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)




params = np.linspace(70,  0.001 , 100)


corset_length = []
accuracy_scores = []
data = []
for dist in tqdm(distances, desc="Distances"):  
    for r in tqdm(params, desc="Parameter r"):
        Coreset, Weights , index = parametric_filter(np.array(X_train), r , dist)
        corset_length.append(len(Coreset))
        
        
        Coreset_features = np.array(X_train)[index]
        Coreset_lables = np.array(y_train)[index]
        
        svc = SVC(kernel='linear', C=1.0, random_state=42) 
        
        try:
            svc.fit(Coreset_features, Coreset_lables)

            y_pred = svc.predict(X_test)

            score =  accuracy_score(y_test, y_pred)

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
 
 
with open("coreset_data.json", "w") as outfile:
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
plt.savefig('Accuracy Score vs. Parameter.png')

# Plot Corset Length vs. Parameter for each distance metric
plt.figure(figsize=(10, 6))
for i, dist in enumerate(distances):
    plt.plot(params, corset_length[i * len(params):(i + 1) * len(params) ], label=dist)
plt.title('Corset Length vs. Parameter')
plt.xlabel('Parameter')
plt.ylabel('Corset Length')
plt.legend()
plt.grid(True)
plt.savefig('Corset Length vs. Parameter.png')