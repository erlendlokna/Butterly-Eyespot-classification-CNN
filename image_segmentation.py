import sklearn.feature_extraction.image as sk
import numpy as np

def segment_image(model, image):
    patches = sk.extract_patches_2d(image, (5, 5))
    patches = np.reshape(patches, (len(patches), 75))
    print(patches.shape)
    #predictions = predict(cnn, np.reshape(patches, (patches.shape[0] ,5, 5, 3, 1)))
    predictions = model.predict(patches)
    
    h = w = int(len(predictions)**0.5)

    img = np.zeros((w, h, 3))

    for i in range(h):
        for j in range(w):
            value = predictions[i*w + j]
            if(value == 0): img[i, j] = [0,0,0]
            elif(value==1): img[i,j]=[0,0,0]
            else: img[i,j]=[250, 250, 250]
    
    return img
