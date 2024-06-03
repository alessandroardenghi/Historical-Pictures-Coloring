import os 
import cv2
import numpy as np

def all_preprocessing(data_dir, image_path, height, width):

  filepath = os.path.join(data_dir, image_path)
  img = cv2.imread(filepath)[:,:,::-1]
  image = cv2.resize(img, (height,width))
  lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  #Converting Image from RGB to LAB
  L, A, B = cv2.split(lab_image)

  X = np.array([L])
  Y = np.array([A,B])

  # The pretrained model accepts 3 channels as input, so we cast X to the correct shape
  X = np.squeeze(X)
  X = np.stack([X] * 3, axis=-1)

  # Casting y to the correct shape
  Y = np.transpose(Y, (1, 2, 0))

  return X, Y

def data_generator(data_dir, batch_size, start, size, height, width):

    num_samples = size
    start_index = 0

    while True:
        X_batch = []
        Y_batch = []
        valid_extensions = ('.jpg', '.png', '.jpeg')

        end_index = min(start_index + batch_size, num_samples)

        # Get a batch of image paths
        all_files = os.listdir(data_dir)
        image_files = [file for file in all_files if file.lower().endswith(valid_extensions)][start:start+size]
        batch_paths = image_files[start_index:end_index]


        # Load and preprocess the images
        for image_path in batch_paths:

            X, Y = all_preprocessing(data_dir, image_path, height, width)
            
            # Append to the batch
            X_batch.append(X)
            Y_batch.append(Y)

        # Convert to NumPy arrays and yield the batch
        X_batch = np.array(X_batch)
        Y_batch = np.array(Y_batch)
        start_index = end_index % num_samples
        yield X_batch, Y_batch
        
def all_preprocessing_crossentropy(data_dir, image_path, height, width, inverse_index_map, w_vector, bin_size = 10):

  filepath = os.path.join(data_dir, image_path)
  img = cv2.imread(filepath)[:,:,::-1]
  image = cv2.resize(img, (height,width))
  lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
  L, A, B = cv2.split(lab_image)

  X = np.array([L])
  Y = np.array([A,B])
  
  ##processing on X
  X = np.squeeze(X)
  X = np.stack([X] * 3, axis=-1)

  ##preprocessing on Y
  Y = np.transpose(Y, (1, 2, 0))
  
  y_final = np.zeros((height, width, 1))
  for row in range(len(Y)):
    for column in range(len(Y[0])):
        x_bin = int(np.floor(Y[row, column, 0] / bin_size))
        y_bin = int(np.floor(Y[row, column, 1] / bin_size))
        y_final[row, column, 0] = int(inverse_index_map[(y_bin, x_bin)])

  
  block_w = np.zeros_like(y_final)
  for i in range(128):
      for j in range(128):
          block_w[i, j, 0] = w_vector[int(y_final[i, j, 0])]


  return X, y_final , block_w

def weighted_data_generator(data_dir, batch_size, start, size, height, width, inverse_index_map, w_vector):

    num_samples = size
    start_index = 0

    while True:
        X_batch = []
        Y_batch = []
        batch_weights = [] 
        valid_extensions = ('.jpg', '.png', '.jpeg')

        end_index = min(start_index + batch_size, num_samples)

        # Get a batch of image paths
        all_files = os.listdir(data_dir)
        image_files = [file for file in all_files if file.lower().endswith(valid_extensions)][start:start+size]
        batch_paths = image_files[start_index:end_index]


        # Load and preprocess the images
        for image_path in batch_paths:
            X, Y , block_w = all_preprocessing_crossentropy(data_dir, image_path, height, width, inverse_index_map, w_vector)

            # Append to the batch
            X_batch.append(X)
            Y_batch.append(Y)
            batch_weights.append(block_w)

        # Convert to NumPy arrays and yield the batch
        X_batch = np.array(X_batch)
        Y_batch = np.array(Y_batch, dtype = int)
        batch_weights = np.array(batch_weights)
        start_index = end_index % num_samples

        yield X_batch, Y_batch , batch_weights