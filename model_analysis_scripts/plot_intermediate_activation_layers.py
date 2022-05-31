#View intermediate layer activations and softmax outputs, plot overlay on satellite image

import keract
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import data, img_as_float
from skimage import exposure

#number of images
num_img = X_test.shape[0]

#output colormap
cmap = plt.cm.coolwarm
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

index = -1

y_pred = model.predict(X_test)

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

for i in range(num_img): 
    print(i)
    index += 1
    image = X_test[i:i+1, :, :, :] #satellite image
    
    #extract rgb color bands from image
    r_band = image[0, :, :, 2]
    g_band = image[0, :, :, 1]
    b_band = image[0, :, :, 0]
    
    #normalize bands    
    redn = normalize(r_band)
    greenn = normalize(g_band)
    bluen = normalize(b_band)
    
    #stack rgb bands 
    rgb_image = np.dstack((redn, greenn, bluen))
    rgb_image = exposure.adjust_log(rgb_image, 1.5)
    
    plt.imshow(rgb_image)
    
    
    #activations = keract.get_activations(model, x = image, output_format="simple", nested=False, auto_compile=True)
    #to index activations
    #activation_list = list(activations.items())
    #view last activation, softmax layer
    #softmax = activation_list[-1]
    #softmax = np.asarray(softmax[1])
    #y_pred = model.predict(X_val)
    #view softmax layers
    w=50
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 6
    rows = 1
    class_names = ["background", "vegetation", "buildings", "roads", "water", "agriculture", "spurge"]
    for i in range(1, columns*rows +1):
    #img = np.random.randint(10, size=(h,w))
        #img = softmax[0, :, :, i]
        img = y_pred[index, :, :, i]
        x = fig.add_subplot(rows, columns, i)
        x.title.set_text(class_names[i])
        plt.imshow(rgb_image)
        plt.imshow(img, alpha=0.7, cmap=my_cmap)
        plt.axis('off')
    plt.show()
    fig.savefig(r"C:/Users/.../prediction_scenes/scene{index}.png".format(index=index), dpi=200) #edit file path
    
