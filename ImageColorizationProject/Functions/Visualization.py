import numpy as np
import matplotlib.pyplot as plt
import cv2

def MSE_plot_predicted_image(original_images, predictions):
    
    final_images = []
    for i in range(len((original_images[0]))): 
        final_images.append([])
        # Combine X (grayscale) and predicted values (AB channels)
        L = np.array(original_images[0][i][:, :, 0], dtype = np.uint8)
        A_true = np.array(original_images[1][i][:, :, 0], dtype = np.uint8)
        B_true = np.array(original_images[1][i][:, :, 1], dtype = np.uint8)
        true_image = cv2.merge([L, A_true, B_true])
        true_image = cv2.cvtColor(true_image, cv2.COLOR_LAB2RGB)
        A_pred = np.array(predictions[i, :, :, 0], dtype = np.uint8)
        B_pred = np.array(predictions[i, :, :, 1], dtype = np.uint8)
        reconstructed_image = cv2.merge([L, A_pred, B_pred])
        rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2RGB)
        final_images[i].append(true_image)
        final_images[i].append(L)
        final_images[i].append(rgb_image)

    # Plot the images
    fig, axes = plt.subplots(len(original_images[0]), 3, figsize=(14, 14))
    for i in range(len(original_images[0])):
        axes[i, 0].imshow(final_images[i][0])
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(final_images[i][1], cmap='gray')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(final_images[i][2])
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def MSE_plot_predicted_image_historical(image, X, predictions):
   
    # Combine X (grayscale) and predicted values (AB channels)
    L = np.array(X[:, :, 0], dtype = np.uint8)
    A = np.array(predictions[ :, :, 0], dtype = np.uint8)
    B = np.array(predictions[ :, :, 1], dtype = np.uint8)
    reconstructed_image = cv2.merge([L, A, B])
    rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2RGB)
        
    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_image)
    axes[1].set_title('Image with Predicted Values')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

    return fig


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# FUNCTION TO MODIFY THE PREDICTED DISTRIBUTION ACCORDING TO T
def f_T(z, T):
    return np.exp((np.log(z)/T))/np.sum(np.exp((np.log(z)/T)))

def visualize_predictions(original_images, predictions, T, height, width, index_map):
    
    final_images = []
    color_vector = np.zeros((243, 2))
    for i in range(len(color_vector)):
        color_vector[i, 0] = index_map[i][0] * 10 + 5
        color_vector[i, 1] = index_map[i][1] * 10 + 5
    
    for i in range(len((original_images[0]))): 
        final_images.append([])
        # Combine X (grayscale) and predicted values (AB channels)
        L = np.array(original_images[0][i][:, :, 0], dtype = np.uint8)
        A_true = np.array(original_images[1][i][:, :, 0], dtype = np.uint8)
        B_true = np.array(original_images[1][i][:, :, 1], dtype = np.uint8)
        true_image = cv2.merge([L, A_true, B_true])
        true_image = cv2.cvtColor(true_image, cv2.COLOR_LAB2RGB)
        b_channel = np.zeros((height, width), dtype=np.uint8)
        a_channel = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            for column in range(width):
                Z_pred = softmax(predictions[i, row, column])
                a = f_T(Z_pred, T)
                a_value = np.sum(a * color_vector[:, 1])
                b_value = np.sum(a * color_vector[:, 0])
                b_channel[row, column] = b_value
                a_channel[row, column] = a_value
        
        reconstructed_image = cv2.merge([L, a_channel, b_channel])
        rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2RGB)
        final_images[i].append(true_image)
        final_images[i].append(L)
        final_images[i].append(rgb_image)

    # Plot the images
    fig, axes = plt.subplots(len(original_images[0]), 3, figsize=(20, 20))
    for i in range(len(original_images[0])):
        axes[i, 0].imshow(final_images[i][0])
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(final_images[i][1], cmap='gray')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(final_images[i][2])
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)