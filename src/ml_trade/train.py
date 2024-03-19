from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import test_train_split
from sklearn.preprocessing import StandardScaler

def validation(data):
    increment_length=32 # value by which the iterator is incremented in eaach step
                        # data length/increment length= no of test-train saamples
    partition_length=40 # two business months of the dataset
     
    # Models to be used
    rfc=RandomForestClassifier() # for training 
    xtc=ExtraTreesClassifier() # for training 
    scaler=StandardScaler() # for feature scaling
    
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
        
        #Obtain input data features and target values (Label Generation)
        label=df_train['pred']
        syn_features=[x for x in df_train.columns if x!='pred']
        input=df_train[syn_features]

        # Make the mean of features 0 and standard deviation 1 (Data scaling)
        input=scaler.fit_transform(input)

        # Split the data into training and testing sets
        input_train, input_text, label_train, label_test=test_train_split(input,label,test_size=0.2, shuffle=False)






