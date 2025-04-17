# Pneumonia_CNN

## Summary
This project objective was to detertmine whether computer vision could accurately diagnose patients with pneumonia solely based on X-ray images using a convolutional neural network (CNN). The key differentiator between normal and pneumonia X-rays is lobar consolidation-a key indicator for determining pneumonia. The X-ray images were preprocessed and data labels were created before training the model. In conclusion, the CNN model was able to able to accurately predict which patients had pneumonia with an accuracy rate between 75% to 80%. The data orginated from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.


## Objective

This experiment aims to determine whether machine learning can differentiate the differences between normal patients and patients whose lungs have been infected with pneumonia with high accuracy. If successful, diagnosing early cases of pneumonia may become faster and more efficient.


## Data Visualization

![__results___3_1](https://github.com/user-attachments/assets/a9fe2597-9fdf-4c3a-b9da-f23200cde02a)
![__results___4_1](https://github.com/user-attachments/assets/355c239d-3a3d-439a-ae4f-59aa79cebd22)


The main differentiation between these lungs is that pneumonia-infected lungs contain Lobar consolidation-a condition where fluid such as blood enters airspaces limiting respiration. It is diagnosed by using an X-ray to identify a blob of a darker shade not visible in normal lungs. Lobar consolidation can occur in any part of the lungs causing a diagnosis to become more complicated than other diseases. 

## Data Preprocessing

Prior to creating the CNN model, all images underwent a transformation to standarize values such as color channels and image dimensions to prevent errors in training. The orginal data conatained very few images for validation, so a portion of the training dataset was used as validation data instead to prevent overfitting and biases in the model. 

## Results
![__results___23_0](https://github.com/user-attachments/assets/4cd0b6ee-70ca-4db0-8723-53e83ac03948)

The more the model cycles through the data, the more the loss rate decreases. 


![__results___24_0](https://github.com/user-attachments/assets/b27d6766-db91-4a14-b2ce-bd3763ffad12)

The accuracy also increases as the model continues training, The loss rate and accuracy are inversely proportional of each other. 

When the model is trained on data foreign to it the accuracy range decreases to between 75% to 80%. This is an understandable tradeoff proving while not perfect the model is not biased or overfitted to the training data. 

##Conclusion
A convolutional neural network can identify pneumonia at high accuracy, however, there are risks in letting it do so. In healthcare, high accuracy scores are not sufficient especially when diagnosing. The possibility of even 10% of people getting misdiagnosed can have devastating consequences causing people who have pneumonia to be treated as normal. Even diagnosing a normal person with pneumonia can cause harm by causing them to take unnecessary medications. 

Another consideration is that X-rays aren't the only factor in diagnosing pneumonia. A pulse oximeter is used to detect oxygen levels in blood which may be limited by pneumonia. Even blood tests are taken to observe if the body is fighting an infection. 

This isn't to say that computer vision can't improve or has no place in healthcare. High quality and quantity of images can allow for better feature extraction to improve a model's success rate. Stronger computation power could also be another lever to provide superior results. Finally, computer vison models would be better suited for lower risk conditions were false positives or negatives don't lead to disastrous results. 

