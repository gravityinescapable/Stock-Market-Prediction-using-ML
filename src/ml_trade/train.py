from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap
import pandas as pd
import numpy as np

def validation(data):
    increment_length=32 # value by which the iterator is incremented in each step
                        # data length/increment length= no of test-train samples
    partition_length=40 # two business months of the dataset
     
    # Models to be used
    rfc=RandomForestClassifier() # for training 
    xtc=ExtraTreesClassifier() # for training 
    scaler=StandardScaler() # for feature scaling
    
    # Lists to store the results
    rfc_results=[]
    xtc_results=[]

    # Initialise the counter to 0
    i=0

    while True:
        
        # Partition the data into chunks of size partition_length
        # For the window sliding approach, the window is rolled/incremented by increment_length
        df_train = data.iloc[i * increment_length : (i * increment_length) + partition_length]
        print(f"current_window: {i * increment_length, (i * partition_length) + increment_length}")
        i += 1
        # Termination condition, the data left is not enough to form a new partition
        if len(df_train) < partition_length:
            break
        
        # Obtain input data features and target values (Label Generation)
        label=df_train['pred']
        syn_features=[x for x in df_train.columns if x!='pred']
        input=df_train[syn_features]

        # Make the mean of features 0 and standard deviation 1 (Data scaling)
        input=scaler.fit_transform(input)

        # Split the data into training and testing sets
        input_train, input_test, label_train, label_test=train_test_split(input,label,test_size=0.2, shuffle=False)

        # Fit the models
        rfc.fit(input_train,label_train)
        xtc.fit(input_train,label_train)

        # Get the predictions
        rfc_prediction=rfc.predict(input_test)
        xtc_prediction=xtc.predict(input_test)

        # Get the accuracy
        rfc_accuracy=accuracy_score(label_test,rfc_prediction)
        xtc_accuracy=accuracy_score(label_test,xtc_prediction)

        print(f"RFC{rfc_accuracy}, XTC{xtc_accuracy}")

        # Append the results
        rfc_results.append(rfc_accuracy)
        xtc_results.append(xtc_accuracy)
        
    print('RFC Accuracy='+str(sum(rfc_results)/len(rfc_results)))
    print('XTC Accuracy='+str(sum(xtc_results)/len(xtc_results)))
    
    # Shapely Additive Explanation for Random Forest Classifier
    explainer_rfc=shap.Explainer(rfc,input_train)
    shap_values_rfc=explainer_rfc.shap_values(input_test)  

    # Plot the results
    shap.summary_plot(shap_values_rfc, input_test, max_display=10)
   
    # Shapely Additive Explanation for Extra Trees Classifier
    explainer_xtc=shap.Explainer(xtc,input_train)
    shap_values_xtc=explainer_xtc.shap_values(input_test)  

    # Plot the results
    shap.summary_plot(shap_values_xtc, input_test, max_display=10)
     
    return rfc,xtc

       