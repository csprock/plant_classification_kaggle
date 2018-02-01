
from sklearn.metrics import accuracy_score

def class_predictions(class_predictions, class_labels):
    indices = np.argmax(class_predictions, axis = 1)
    labels = []
    for i in indices:
        labels.append(class_labels[i])    
        
    return np.array(labels)




def predict_ensamble(X_data, regular_model, binary_model, full_classes, binary_classes):
    
    predReg = class_predictions(regular_model.predict(X_data), full_classes)
    
    binary_subset = np.array([i for i,j in enumerate(predReg) if j in binary_classes])
    
    Xb_data = X_data[binary_subset, :,:,:]
    predBinary = class_predictions(binary_model.predict(Xb_data), binary_classes)
    
    predReg[binary_subset] = predBinary
    return list(predReg)




predEnsamble = predict_ensamble(X_valid, reg_model, binary_model, classes, ['Black-grass','Loose Silky-bent'])

#accuracy_score(Y_test_label, list(predEnsamble))




output = pd.DataFrame(list(zip(test_set_names, predEnsamble)))
output.to_csv('C:/Users/csprock/Documents/Projects/Kaggle/Plant_Classification/test_output.csv')