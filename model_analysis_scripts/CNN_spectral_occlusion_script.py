'''
These functions apply spectral noise to satellite imagery
using sampled spectral reflectance data from leafy spurge and other vegetation classes
After selecting a sample satellite image, the mean spectral reflectance is increased systematically
and changes in the model's predictions for leafy spurge are output
'''


# spectral occlusion in planetscope CNN models
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import data, img_as_float
from skimage import exposure

#load trained CNN model
model = load_model(r'C:/users/.../my_model.h5')
model_name = "my_model"

#select imagery only from the testing set
split = 'test'

#set path to dataset
path_to_training_data = 'C:/users/.../training_data/'

#define the names for each AOI used in the study
aoi_list = ["aoi0", "aoi1", "aoi2", "aoi3", "aoi4", "aoi5"]

#Define the 12 dates for each of the six study AOIs
#Date names correspond to folder names within directory
aoi_dates_list = [['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July032020','July122020','July192020'],
                  ['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July122020','July152020','July182020'],
                  ['May152020','May192020','June012020','June032020','June052020','June142020','June162020','June272020','July102020','July122020','July182020','July242020'],
                  ['May082020','May122020','May152020','May202020','June032020','June052020','June072020','June172020','June252020','July032020','July122020','July302020'],
                  ['May122020','May202020','June032020','June052020','June072020','June122020','June172020','June252020','July022020','July122020','July182020','July302020'],
                  ['May122020','May202020','June012020','June062020','June172020','June252020','July082020','July102020','July122020','July192020','July242020','July302020']]
            

#function to select Planetscope satellite imagery, returns X_val (image) and corresponding Y_val (mask) tensors
def get_val_or_test_data(split='test'):
    """
    Load validation data for one or several dates in the suitable form
    for the baseline CNN.
    """
    print(f'Loading and processing "{split}" subset data...')

    # define path to training data
    path_to_training_data = 'F:/.../data/processed/training_data/'
    aoi_list = ["aoi0", "aoi1", "aoi2", "aoi3", "aoi4", "aoi5"]
    #only 12 dates for a time period
    aoi_dates_list = [['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July032020','July122020','July192020'],
                      ['May082020','May122020','May202020','June012020','June052020','June072020','June122020','June172020','June252020','July122020','July152020','July182020'],
                      ['May152020','May192020','June012020','June032020','June052020','June142020','June162020','June272020','July102020','July122020','July182020','July242020'],
                      ['May082020','May122020','May152020','May202020','June032020','June052020','June072020','June172020','June252020','July032020','July122020','July302020'],
                      ['May122020','May202020','June032020','June052020','June072020','June122020','June172020','June252020','July022020','July122020','July182020','July302020'],
                      ['May122020','May202020','June012020','June062020','June172020','June252020','July082020','July102020','July122020','July192020','July242020','July302020']]

    files=[] #files[0] is aoi0:may032020, files[1] is aoi0:may082020...
    for a in range(len(aoi_dates_list)):
        for d in range(len(aoi_dates_list[a])):
            ps_folder_path = path_to_training_data+f'{aoi_list[a]}/{aoi_dates_list[a][d]}_data/{aoi_dates_list[a][d]}_{split}_tiles_(128x128)'
            files.append([x for x in glob(ps_folder_path+'/*.tif')])
    #len(files) yields 72 folders, and #files[0] is aoi0, may 8, list of 26 tif files
    
    files_list = []
    for sublist in files:
        for item in sublist:
            files_list.append(item)
        
    n_tiles = len(files_list) #2328 total test tiles

    image_data = np.empty((n_tiles, 128, 128, 5))
    
    for j in range(n_tiles):
        image_data[j, :, :, :] = np.array(imread(files_list[j], plugin="tifffile"), dtype=float)     
    
    X_val = image_data[:, :, :, :4]
    Y_val = image_data[:, :, :, 4]

    return X_val, Y_val



#obtain validation images and masks
X_val, Y_val = get_val_or_test_data(split="test") #validation data shape (index, 128, 128, 4)

#selct only single image and mask pair for spectral data manipulation experiments
normal_image = X_val[2:3, :, :, :]
normal_image_mask = Y_val[2:3, :, :]


#Normalize satellite image bands and plot selected image in natural color
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

num_img = 1
for i in range(num_img):     
    #extract rgb color bands from image
    r_band = normal_image[0, :, :, 2]
    g_band = normal_image[0, :, :, 1]
    b_band = normal_image[0, :, :, 0]
    
    #normalize bands    
    redn = normalize(r_band)
    greenn = normalize(g_band)
    bluen = normalize(b_band)
    
    #stack rgb bands 
    rgb_image = np.dstack((redn, greenn, bluen))
    rgb_image = exposure.adjust_log(rgb_image, 1.0)
    
    plt.imshow(rgb_image)
    

#predict image without spectral occlusions
normal_image_pred = model.predict(normal_image)

#copy original image, apply spectral manipualtions to occlude spectral information in satellite bands
occlusion_image_noband = normal_image.copy()


#occlude spectral band sampled from distribution of existing spectral band
import scipy.stats as stats

spurge_softmax = np.empty(shape = (120, 2))

band_mean = 0 #set the spectral reflectance mean for sampling to zero 
for s in range(120): #for loop iterates through range of spectral reflectance samples
    band_mean += 100 #every iteration, increase the spectral reflectance mean by 100
    occlusion_image_noband = normal_image.copy()
    #random box occlusion
    #function to randomly erase a portion of an image for occlusion experiments
    def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=6000, pixel_level=True, pixellevel=True):
        def eraser(input_img):
            if input_img.ndim == 3:
                img_h, img_w, img_c = input_img.shape
            elif input_img.ndim == 2:
                img_h, img_w = input_img.shape
    
            p_1 = np.random.rand()
    
            if p_1 > p:
                return input_img
    
            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                #w = int(np.sqrt(s / r))
                #h = int(np.sqrt(s * r))
                w = 50
                h = 50
                #left = np.random.randint(0, img_w)
                #top = np.random.randint(0, img_h)
                left = 0
                top = 0
    
                if left + w <= img_w and top + h <= img_h:
                    break
    
            if pixellevel:
                if input_img.ndim == 3:
                    #c = np.random.uniform(v_l, v_h, (h, w, img_c))
                    #c = np.random.normal(loc=band_mean, scale=band_std, size=(h, w))
					
					#several types of occlusions, uncomment to run
					
                    #leafy spurge band averages and stds sampled from Planetscope imagery
                    band0_ls = np.random.normal(loc=470.1082, scale=121.9494, size=(h, w, 1))
                    band1_ls = np.random.normal(loc=832.1973, scale=139.1215, size=(h, w, 1))
                    band2_ls = np.random.normal(loc=605.5022, scale=210.9809, size=(h, w, 1))
                    band3_ls = np.random.normal(loc=3803.9796, scale=584.7553, size=(h, w, 1))
                                    
                    #replace leafy spurge band mean with set value in increments of 100 
                    #band0_ls = np.random.normal(loc=band_mean, scale=121.9494, size=(h, w, 1))
                    #band1_ls = np.random.normal(loc=band_mean, scale=139.1215, size=(h, w, 1))
                    #band2_ls = np.random.normal(loc=band_mean, scale=210.9810, size=(h, w, 1))
                    #band3_ls = np.random.normal(loc=band_mean, scale=584.7553, size=(h, w, 1))

                    #vegetation band averages and stds
                    #band0_veg = np.random.normal(loc=417.8549, scale=111.2987, size=(h, w, 1))
                    #band1_veg = np.random.normal(loc=714.2001, scale=150.8813, size=(h, w, 1))
                    #band2_veg = np.random.normal(loc=522.0829, scale=185.2645, size=(h, w, 1))
                    #band3_veg = np.random.normal(loc=3658.0920, scale=640.5959, size=(h, w, 1))
                    
                    #replace vegetation band mean with set value in increments of 100
                    #band0_veg = np.random.normal(loc=band_mean, scale=111.2987, size=(h, w, 1))
                    #band1_veg = np.random.normal(loc=band_mean, scale=150.8813, size=(h, w, 1))
                    #band2_veg = np.random.normal(loc=band_mean, scale=185.26451, size=(h, w, 1))
                    #band3_veg = np.random.normal(loc=band_mean, scale=640.5959, size=(h, w, 1))
                    
                    c = np.dstack((band0_ls, band1_ls, band2_ls, band3_ls))
                
                    #c = np.dstack((band0_veg, band1_veg, band2_veg, band3_veg))
                    
                if input_img.ndim == 2:
                    c = np.random.uniform(v_l, v_h, (h, w))
            else:
                c = np.random.uniform(v_l, v_h)
    
            input_img[top:top + h, left:left + w] = c
    
            return input_img
    
        return eraser
    
 
    #create instance of eraser
    eraser = get_random_eraser(p=1, s_l=0.1, s_h=0.3, r_1=0.3, r_2=1/0.3, v_l=1, v_h=6000, pixel_level=True)
    
    #erase random area from image, replace erased data with new spectral manipulations
    erased_image = eraser(occlusion_image_noband[0, :, :, :])
    erased_image = np.expand_dims(erased_image, 0)
    
    #predict model with occluded input data
    occlusion_image_noband_pred = model.predict(erased_image)

    spurge_softmax[s, 0] = band_mean
    spurge_softmax[s, 1] = occlusion_image_noband_pred[0, :, :, 1].mean()


#plot interaction between spectral bands and leafy spurge softmax values
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('softmax')
ax1.set_xlabel('band 3 mean')
plt.scatter(spurge_softmax[:, 0], spurge_softmax[:, 1])
plt.show()


#visualize model predictions via softmax
w = 20
h = 10
fig = plt.figure(figsize=(20, 10))
columns = 8
rows = 2

#Regular Prediction
#plot image
sat_img = fig.add_subplot(1, 8, 1)
plt.imshow(rgb_image)
sat_img.title.set_text("Sat Img")
plt.axis('off')

#plot mask
mask = fig.add_subplot(1, 8, 2)
plt.imshow(normal_image_mask[0, :, :])
mask.title.set_text("Mask")
plt.axis('off')

#plot model predictions
veg = fig.add_subplot(1, 8, 3)
plt.imshow(occlusion_image_noband_pred[0, :, :, 1])
veg.title.set_text("Veg")
plt.axis('off')

buildings = fig.add_subplot(1, 8, 4)
plt.imshow(occlusion_image_noband_pred[0, :, :, 2])
buildings.title.set_text("Buildings")
plt.axis('off')

roads = fig.add_subplot(1, 8, 5)
plt.imshow(occlusion_image_noband_pred[0, :, :, 3])
roads.title.set_text("Roads")
plt.axis('off')

water = fig.add_subplot(1, 8, 6)
plt.imshow(occlusion_image_noband_pred[0, :, :, 4])
water.title.set_text("Water")
plt.axis('off')

ag = fig.add_subplot(1, 8, 7)
plt.imshow(occlusion_image_noband_pred[0, :, :, 5])
ag.title.set_text("Ag")
plt.axis('off')

spurge = fig.add_subplot(1, 8, 8)
plt.imshow(occlusion_image_noband_pred[0, :, :, 6])
spurge.title.set_text("Spurge")
plt.axis('off')

sat_img = fig.add_subplot(2, 8, 1)
plt.imshow(occlusion_image_noband[0, :, :, 0])
sat_img.title.set_text("Blue")
plt.axis('off')

sat_img = fig.add_subplot(2, 8, 2)
plt.imshow(occlusion_image_noband[0, :, :, 1])
sat_img.title.set_text("Green")
plt.axis('off')

sat_img = fig.add_subplot(2, 8, 3)
plt.imshow(occlusion_image_noband[0, :, :, 2])
sat_img.title.set_text("Red")
plt.axis('off')

sat_img = fig.add_subplot(2, 8, 4)
plt.imshow(occlusion_image_noband[0, :, :, 3])
sat_img.title.set_text("NIR")
plt.axis('off')

