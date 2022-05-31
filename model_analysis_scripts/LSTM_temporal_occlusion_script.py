'''
These functions apply gaussian noise to a timeseries of satellite imagery
then predict on the manipulated timeseries
'''



#Load trained LSTM model
lstm_model = load_model(r'C:/users/.../my_model.h5')
model_name = "my_model"

#set path to dataset
path_to_training_data = 'C:/users/.../training_data/'
#select imagery only from the testing set
split="test"

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

#analyses contains 12 timesteps
timesteps = 12 

files=[] #files[0] is aoi0:may032020, files[1] is aoi0:may082020...
for a in range(len(aoi_dates_list)): #nested for loops iterate through each aoi_list and each aoi_dates_list to get all file names in testing dataset
    for d in range(len(aoi_dates_list[a])):
        ps_folder_path = path_to_training_data+f'{aoi_list[a]}/{aoi_dates_list[a][d]}_data/{aoi_dates_list[a][d]}_{split}_tiles_(128x128)'
        files.append([x for x in glob(ps_folder_path+'/*.tif')])

n_features = 4 #num satellite bands
timeseries_len = 12 #num timesteps
batch_size = 10 #num of images to draw in the loop below (can be any index, typically set to 10 to save memory)

#initialize empty data containers for X [image], Y [mask], and one-hot Y [encoded mask]
X_val = np.empty((batch_size, timeseries_len, 128, 128, n_features))
Y_val = np.empty((batch_size, timeseries_len, 128, 128))



#Custom for-loop that randomly draws a time-series of satellite images of batch_size
#Loop randomly selects an AOI, then through number of batches appends images to X_val and Y_val of size (batch_size, 12, 128, 128, 4)
for b in range(batch_size):
    #randomly select AOI
    aoi_random_choice = np.random.randint(0, len(aoi_list), 1)
    selected_aoi = aoi_list[aoi_random_choice[0]]

    ### AOI SUBSET DATES ###

    #After randomly selecting an AOI, define files from the set of 12 dates within that AOI
    if selected_aoi == "aoi0":
        subset_files = files[0:12]
    if selected_aoi == "aoi1":
        subset_files = files[12:24]
    if selected_aoi == "aoi2":
        subset_files = files[24:36]
    if selected_aoi == "aoi3":
        subset_files = files[36:48]
    if selected_aoi == "aoi4":
        subset_files = files[48:60]
    if selected_aoi == "aoi5":
        subset_files = files[60:72]

    #amount of timestep padding (zeros) to add before LSTM input (for testing only, here there's zero padding)
    lstm_padding_amount = timeseries_len - len(subset_files)      
    #print(f"padding {lstm_padding_amount}")

    #from files, randomly select one INDEX to use in time series stack
    file_random_choice = np.random.randint(0, len(subset_files[0]), 1)
    #select time series samples based on filepaths of index file_random_choice
    stacked_file_date_aoi = []
    for k in range(len(aoi_dates_list[aoi_random_choice[0]])):
        #stacked_files_dates[k] is the path for the same image at all timesteps
        stacked_file_date_aoi.append(subset_files[k][file_random_choice[0]])

    batch_X = []
    batch_Y = []

    for j, input_path in enumerate(stacked_file_date_aoi):
        image = np.array(imread(input_path, plugin="tifffile"), dtype=float)
        
        batch_X += [image[:, :, :4]]
        batch_Y += [image[:, :, 4]]
        
    X_val[b, ...] = np.array(batch_X) # shape batch, time, 128, 128, 4
    Y_val[b, ...] = np.array(batch_Y) #shape batch, time, 128, 128



#View intermediate layer activations and softmax outputs, plot overlay on satellite image
#import keract
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import data, img_as_float
from skimage import exposure

#funciton to normalize image to display in natural color 
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

image = X_val[0:1, :, :, :, :] #select only one image from tensor [index, 128, 128, 3, 11] 
mask = Y_val[0:1, :, :, :] #select nly ground truth from tensor [index, 128, 128, 1]


#temporal image occlusion
#copy the origianl image to not overwrite later during image occlusion experiments
occlusion_image = image.copy()

#occlude single spectral band with zero value
image_band0_means = []
image_band1_means = []
image_band2_means = []
image_band3_means = []
for m in range(12): #for each 12 timesteps, in each spectral band, sample the mean of the spectral reflectance
        mean_band0_value = image[:, m, :, :, 0].mean() 
        mean_band1_value = image[:, m,  :, :, 1].mean()
        mean_band2_value = image[:, m,  :, :, 2].mean()
        mean_band3_value = image[:, m,  :, :, 3].mean()
        
        image_band0_means.append(mean_band0_value)
        image_band1_means.append(mean_band1_value)
        image_band2_means.append(mean_band2_value)
        image_band3_means.append(mean_band3_value)
        

image_band0_stds = []
image_band1_stds = []
image_band2_stds = []
image_band3_stds = []
for m in range(12): #for each 12 timesteps, in each spectral band, sample the standard deviation of the spectral reflectance
        std_band0_value = image[:, m,   :, :, 0].std()
        std_band1_value = image[:, m,   :, :, 1].std()
        std_band2_value = image[:, m,   :, :, 2].std()
        std_band3_value = image[:, m,   :, :, 3].std()
        
        image_band0_stds.append(std_band0_value)
        image_band1_stds.append(std_band1_value)
        image_band2_stds.append(std_band2_value)
        image_band3_stds.append(std_band3_value)


#occlude images based on date
occlusion_times = [0, 1, 10, 11] #occlude dates 0, 1, 10, 11

#apply image occlusions by replacing a satellite image spectral band with the mean/std of gaussian random noise sampled from that band
for t in range(12):
   if t in occlusion_times:
       occlusion_image[:, t, :, :, 0] = np.random.normal(loc=image_band0_means[t], scale=image_band0_stds[t], size=(128, 128)) #occlude band 0 by image band 0 mean, std noise
       occlusion_image[:, t, :, :, 1] = np.random.normal(loc=image_band1_means[t], scale=image_band1_stds[t], size=(128, 128)) #occlude band 1 by image band 1 mean, std noise
       occlusion_image[:, t, :, :, 2] = np.random.normal(loc=image_band2_means[t], scale=image_band2_stds[t], size=(128, 128)) #occlude band 2 by image band 2 mean, std noise
       occlusion_image[:, t, :, :, 3] = np.random.normal(loc=image_band3_means[t], scale=image_band3_stds[t], size=(128, 128)) #occlude band 3 by image band 3 mean, std noise


rgb_images = np.empty((12, 128, 128, 3)) #hold rgb image timesteps



#Plot the RGB natural color images for the time series after the image occlusions to verify above occlusions worked as intended
for k in range(timesteps): #subset rgb time-series images
    print(k)
    #extract rgb color bands from image
    #r_band = image[0, k, :, :, 2]
    #g_band = image[0, k, :, :, 1]
    #b_band = image[0, k, :, :, 0]
    
    #extract rgb color bands from occluded image
    r_band = occlusion_image[0, k, :, :, 2]
    g_band = occlusion_image[0, k, :, :, 1]
    b_band = occlusion_image[0, k, :, :, 0]

    #normalize bands    
    redn = normalize(r_band)
    greenn = normalize(g_band)
    bluen = normalize(b_band)

    #stack rgb bands 
    rgb_image = np.dstack((redn, greenn, bluen))
    rgb_image = exposure.adjust_log(rgb_image, 1.2)
    
    rgb_images[k, :, :, :] = rgb_image
   
plt.rc('font', size=24) #plotting
fig = plt.figure(figsize=(30, 10))
columns = 7
rows = 2
timestep = 0
for r in range(1, 3, 1):
    for c in range(1, 7, 1):
            img_timestep = fig.add_subplot(2, 6, timestep+1)
            plt.imshow(rgb_images[timestep, ...])
            img_timestep.title.set_text(f"Time {timestep}")
            timestep += 1
            plt.axis('off')    
        

#Use the LSTM model to predict on the occluded image time series
y_pred = lstm_model.predict(occlusion_image) #predict on time-series of images shape (1, 12, 128, 128, 7)
y_pred_softmax = np.argmax(y_pred, axis=-1) #softmax outputs shape (1, 12, 128, 128)


###
#Plotting Effects of the Temporal Image Manipulations 
###

#display prediction classes single time step
plt.rc('font', size=8)
fig = plt.figure(figsize=(8,1.5)) #width, height
columns = 6
rows = 1
timestep = 0
index = 1
class_names = ["Vegetation", "Buildings", "Roads", "Water", "Agriculture", "Leafy Spurge"]
for c in range(1, 7, 1):
    print(f"column index {c}")
    print(f"row index {r}")
    print(f"timestep {timestep}")
    img_timestep = fig.add_subplot(1, 6, index)
    #set plot title for class
    print(class_names[c-1])
    img_timestep.title.set_text(class_names[c-1])
    plt.imshow(y_pred[0, timestep, :, :, c])
    index += 1
    plt.axis('off')
    

#display softmax prediction timeseries for all classes
fig = plt.figure(figsize=(30, 10))
plt.rc('font', size=24)
columns = 6
rows = 2
timestep = 0
for r in range(1, 3, 1):
    for c in range(1, 7, 1):
            img_timestep = fig.add_subplot(2, 6, timestep+1)
            plt.imshow(y_pred_softmax[0, timestep, :, :])
            img_timestep.title.set_text(f"Time {timestep}")
            timestep += 1
            plt.axis('off')
        


#display prediction classes time series 
plt.rc('font', size=8)
fig = plt.figure(figsize=(8,15)) #width, height
columns = 6
rows = 12
timestep = -1
index = 1
class_names = ["Vegetation", "Buildings", "Roads", "Water", "Agriculture", "Leafy Spurge"]
for r in range(1, 13, 1):
    timestep += 1
    for c in range(1, 7, 1):
        #print(f"column index {c}")
        #print(f"row index {r}")
        #print(f"timestep {timestep}")
        img_timestep = fig.add_subplot(12, 6, index)
        #set plot title for class
        img_timestep.title.set_text(class_names[c-1])
        plt.imshow(y_pred[0, timestep, :, :, c])
        index += 1
        plt.axis('off')


