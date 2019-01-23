import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

print("Prediction starts: \n")


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

testImageNameList = os.listdir("dataset/single_prediction/")

counterOfTrueDetection = 0
counterOfTestedImages = 0

ROOT_DIR = os.getcwd()
weightDirectory = os.path.join(ROOT_DIR, "Weights/")
weight_path, weight_dirs, weight_files = next(os.walk(weightDirectory))

resultFile = open("Results.txt","w")
counter = 0
for weight in weight_files:

    print(weight)
    counter += 1
    if not weight.endswith(".DS_Store"):
        classifier.load_weights(weightDirectory+weight)
        for testImageName in testImageNameList:
            if not testImageName.endswith(".DS_Store"):
                counterOfTestedImages += 1
                test_image = image.load_img('./dataset/single_prediction/'+str(testImageName)+'', target_size = (64, 64))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = classifier.predict(test_image)
                #print(result)

                if result[0][0] == 0:
                    if testImageName[0] == 't':
                        counterOfTrueDetection += 1
                    else:
                        prediction = str(testImageName)+' wrong detected as Tile'
                        #print(prediction)
                elif result[0][0] == 1:
                    if testImageName[0] == 'w':
                        counterOfTrueDetection += 1
                    else:
                        prediction = str(testImageName)+' wrong detected as Window'
                        #print(prediction)
                else:
                    if testImageName[0] == 'g':
                        counterOfTrueDetection += 1
                    else:
                        prediction = str(testImageName)+' wrong detected as Garbage'
                        #print(prediction)
    result = round(100*(counterOfTrueDetection/counterOfTestedImages),2)
    resultFile.write(str(weight)+" "+str(result)+"\n")
"""print("\nStatistics for 100E on Big Dataset: \n")
print("Factor of classification: " +str(counterOfTrueDetection/counterOfTestedImages))
print("Tested images: " +str(counterOfTestedImages))
print("True detected: " +str(counterOfTrueDetection))"""

