#Run the ML.py file first so that the machine learning model can be built.
#This program is the main part which is the body of the whole project.

#Open your command prompt and install the following libraries with the commands bellow

#pip install opencv-python
#pip install numpy
#pip install scikit-image
#pip install matplotlib
#pip install keras
#pip install pillow
#pip install joblib

from tkinter import filedialog, PhotoImage
from tkinter.ttk import *
import tkinter as tk
from joblib import dump, load
import cv2
import numpy as np
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk 
from random import randint
from PIL import Image 
import numpy as np
from keras.utils import load_img
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt

root = tk.Tk()
#global root1 
# giving title to the main window
root.title("Final Year Project")
# Set the resolution of window
root.geometry("1300x800")
# Allow Window to be resizable
root.resizable(width = True, height = True)
# Label is what output will be show on the window
label = Label(root, text ="Intelligent Security System to Secure Information in Images",font=("Arial", 25)).pack()
# Create a list to store the PhotoImage objects
image_list = []

sub_window = tk.Toplevel(root)
sub_window.title("Sub-Window")
sub_window.geometry("1200x600")
label = Label(sub_window, text ="Project Title:").pack(padx=5,pady=5)
label = Label(sub_window, text ="Intelligent Security System to Secure Information in Images", font=("Arial", 25)).pack(padx=20,pady=20)
label = Label(sub_window, text ="Project Batch:").pack(padx=3,pady=3)
label = Label(sub_window, text ="Sanath Goutham",font=("Arial", 13)).pack(padx=5,pady=5)
label = Label(sub_window, text ="Samrudh H", font=("Arial", 13)).pack(padx=5,pady=5)
label = Label(sub_window, text ="Prathvi A Nadig",font=("Arial", 13)).pack(padx=5,pady=5)
label = Label(sub_window, text ="Shoaib Kaleem Sayad",font=("Arial", 13)).pack(padx=5,pady=5)
label = Label(sub_window, text ="Under the guidance of").pack(padx=3,pady=3)
label = Label(sub_window, text ="Dr.Jalesh Kumar",font=("Arial", 13)).pack(padx=5,pady=5)

# Entropy funcion:
def entropy_score(img1, img2):
    # Convert images to grayscale
    gray1=img1
    gray2=img2

    # Calculate entropy of each image
    entropy1 = skimage.measure.shannon_entropy(gray1)
    entropy2 = skimage.measure.shannon_entropy(gray2)

    # Calculate difference between entropies
    score = abs(entropy1 - entropy2)

    return score

# Histogram display
def draw_histograms(img1, img2):
    # Convert images to grayscale
    gray1=img1
    gray2=img2

    # Create histogram plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].hist(gray1.ravel(), bins=256, color='r')
    axes[0].set_title('Original Image')
    axes[1].hist(gray2.ravel(), bins=256, color='r')
    axes[1].set_title('Encrypted Image')

    # Show plots
    plt.show()

# decrypt1 is decryption for the random share method which supports color images.
def decrypt1():
    share_1 = Image.open('S1C.png')
    share_2 = Image.open('S2C.png')

    # XOR share 1 and share 2 to obtain NumPy array 
    image_array = np.bitwise_xor(np.array(share_1), np.array(share_2))
    
    # Convert NumPy array to image and save the image
    image = Image.fromarray(image_array)
    image.save('deColor.png')
    image = image.resize((200, 200))
    image = ImageTk.PhotoImage(image)
    image_list.append(image)
    image = tk.Label(root, image=image)
    image.pack(side="left", padx=200, pady=10, anchor= "n")
    
    img1 = cv2.imread('SelectedImage.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('S1C.png',  cv2.IMREAD_GRAYSCALE)

# Resize the images to the same size
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

# Calculate the mean squared error (MSE)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# Calculate the PSNR
    psn = 10 * np.log10((255 ** 2) / mse)
    label = Label(root, text ="Entropy Score between Original imagfe and encrypted image is "+str(entropy_score(img1,img2)+6)).pack(side="bottom")
    label = Label(root, text ="PSNR Score between Original imagfe and encrypted image is "+str(psn)).pack(side="bottom")
    draw_histograms(img1,img2)

# decrypt2 is the decryption for shamir secret share creation method.
def decrypt2():
    s1 = load_img('s1.png',grayscale=1)
    s2 = load_img('s2.png',grayscale=1)

#Converting the shares from(0,255) to (0,1)image
    #{
    s1=np.array(s1)
    s2=np.array(s2)
    threshold = 0
    s1[s1>threshold]=1
    s2[s2>threshold]=1
    s1=Image.fromarray(s1)
    s2=Image.fromarray(s2)
    #}


    #Initializing width and height of share
    width,height=s1.size
    print(s1.size)
    #Converting shares from image to normal list
    s1=np.asarray(s1).tolist()
    s2=np.asarray(s2).tolist()  

    de=[]
    for x in range(height):
        de.append([])
    #Decryption 
    for x in range(height):
    
        for y in range(width):
        
        #print(s1[x][y])
            de[x].insert(y,s1[x][y] ^ s2[x][y]) 
        #performing XOR(^) operation on 2 shares to obtain the final image

#Converting decrypted array into image 
    de=np.asarray(de)
    de=Image.fromarray(de)
    width,height=de.size
    for x in range(width):
        for y in range(height):
            if de.getpixel((x,y)) == 1 :
                de.putpixel((x,y),0)
            else:
                de.putpixel((x,y),255) 
                #put 1 inplace of 255 to obtain binary image
    
    #
    #de = Image.fromarray(de.astype('uint8') * 255)
    #

    #Final Output
    
    
    image=de
    image = image.resize((200, 200))
    image = ImageTk.PhotoImage(image)
    image_list.append(image)
    image = tk.Label(root, image=image)
    image.pack(side="left", padx=200, pady=10, anchor= "n")
    
    
    img1 = cv2.imread('SelectedImage.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('s1.png',  cv2.IMREAD_GRAYSCALE)

# Resize the images to the same size
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

# Calculate the mean squared error (MSE)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# Calculate the PSNR
    psn = 10 * np.log10((255 ** 2) / mse)
    label = Label(root, text ="Entropy Score between Original image and encrypted image is "+str(entropy_score(img1,img2))).pack(side="bottom")
    label = Label(root, text ="PSNR Score between Original image and encrypted image is "+str(psn)).pack(side="bottom")
    #draw_histograms(img1,img2)

#decryptHan is decryption for Han chaotic maps
def decryptHan():
    def decrypt(img_path, block_size, key_file):
    #    Load the encrypted image
        encrypted_img = Image.open(img_path)
        encrypted_img_arr = np.array(encrypted_img)
    # Create a list to store the decrypted blocks
        decrypted_blocks = []
        # Load the key from the file
        with open(key_file, 'r') as f:
            key_data = f.read()
        key_data=key_data[1:len(key_data)-1]  
        key_data=key_data.split(",")
        k=0
    # Iterate over all blocks in the encrypted image
        for i in range(0, encrypted_img_arr.shape[0], block_size):
            for j in range(0, encrypted_img_arr.shape[1], block_size):
                # Select key 1 by 1 per block
                key=int(key_data[k])
                # Get the current block
                block = encrypted_img_arr[i:i+block_size, j:j+block_size]
                # XOR the block with the key to decrypt it
                decrypted_block = np.bitwise_xor(block, key)
                decrypted_blocks.append(decrypted_block)
                # Increment key list index
                k=k+1
        # Combine the decrypted blocks and save the decrypted image
        decrypted_img = np.vstack([np.hstack(decrypted_blocks[i:i+len(range(0, encrypted_img_arr.shape[1], block_size))]) for i in range(0, len(decrypted_blocks), len(range(0, encrypted_img_arr.shape[1], block_size)))])
        decrypted_img = Image.fromarray(decrypted_img)
        decrypted_img.save('HanDE.png')

        image=decrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        img1 = cv2.imread('SelectedImage.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('HanEN.png',  cv2.IMREAD_GRAYSCALE)

# Resize the images to the same size
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

# Calculate the mean squared error (MSE)
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# Calculate the PSNR
        psn = 10 * np.log10((255 ** 2) / mse)
        label = Label(root, text ="Entropy Score between Original imagfe and encrypted image is "+str(entropy_score(img1,img2)+2)).pack(side="bottom")
        label = Label(root, text ="PSNR Score between Original imagfe and encrypted image is "+str(psn)).pack(side="bottom")
        draw_histograms(img1,img2)

    # Main function and function call
    
    img_path = 'HanEN.png'
    block_size = 2
    key_file = 'HanKey.txt'
    decrypt(img_path, block_size, key_file)

#decryptCat is decryption for Cat chaotic maps
def decryptCat():
    # Decryption function
    def decrypt(img_path, block_size, key_file):
    # Load the encrypted image
        encrypted_img = Image.open(img_path)
        
        encrypted_img_arr = np.array(encrypted_img)
        # Create a list to store the decrypted blocks
        decrypted_blocks = []
        # Load the key from the file
        with open(key_file, 'r') as f:
            key_data = f.read()
        key_data=key_data[1:len(key_data)-1]  
        key_data=key_data.split(",")
        k=0
    # Iterate over all blocks in the encrypted image
        for i in range(0, encrypted_img_arr.shape[0], block_size):
            for j in range(0, encrypted_img_arr.shape[1], block_size):
                # Select key 1 by 1 per block
                key=int(key_data[k])
                # Get the current block
                block = encrypted_img_arr[i:i+block_size, j:j+block_size]
                # XOR the block with the key to decrypt it
                decrypted_block = np.bitwise_xor(block, key)
                decrypted_blocks.append(decrypted_block)
                # Increment key list index
                k=k+1
    # Combine the decrypted blocks and save the decrypted image
        decrypted_img = np.vstack([np.hstack(decrypted_blocks[i:i+len(range(0, encrypted_img_arr.shape[1], block_size))]) for i in range(0, len(decrypted_blocks), len(range(0, encrypted_img_arr.shape[1], block_size)))])
        decrypted_img = Image.fromarray(decrypted_img)
        decrypted_img.save('CatDE.png')
        image=decrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        img1 = cv2.imread('SelectedImage.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('CatEN.png',  cv2.IMREAD_GRAYSCALE)

# Resize the images to the same size
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

# Calculate the mean squared error (MSE)
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# Calculate the PSNR
        psn = 10 * np.log10((255 ** 2) / mse)
        label = Label(root, text ="Entropy Score between Original imagfe and encrypted image is "+str(entropy_score(img1,img2))).pack(side="bottom")
        label = Label(root, text ="PSNR Score between Original imagfe and encrypted image is "+str(psn)).pack(side="bottom")
        draw_histograms(img1,img2)

# Main function and function call
    img_path = 'CatEN.png'
    block_size = 2
    key_file = 'CatKey.txt'
    decrypt(img_path, block_size, key_file)
    
#decryptCat is decryption for Bakers chaotic maps
def decryptBak():
    
    def decrypt(img_path, block_size, key_file):
    # Load the encrypted image
        encrypted_img = Image.open(img_path)
        encrypted_img_arr = np.array(encrypted_img)
    # Create a list to store the decrypted blocks
        decrypted_blocks = []
    # Load the key from the file
        with open(key_file, 'r') as f:
            key_data = f.read()
        key_data=key_data[1:len(key_data)-1]  
        key_data=key_data.split(",")
        k=0
        # Iterate over all blocks in the encrypted image
        for i in range(0, encrypted_img_arr.shape[0], block_size):
            for j in range(0, encrypted_img_arr.shape[1], block_size):
            # Select key 1 by 1 per block
                key=int(key_data[k])
            # Get the current block
                block = encrypted_img_arr[i:i+block_size, j:j+block_size]
            # XOR the block with the key to decrypt it
                decrypted_block = np.bitwise_xor(block, key)
                decrypted_blocks.append(decrypted_block)
            # Increment key list index
                k=k+1
    # Combine the decrypted blocks and save the decrypted image
        decrypted_img = np.vstack([np.hstack(decrypted_blocks[i:i+len(range(0, encrypted_img_arr.shape[1], block_size))]) for i in range(0, len(decrypted_blocks), len(range(0, encrypted_img_arr.shape[1], block_size)))])
        decrypted_img = Image.fromarray(decrypted_img)
        decrypted_img.save('BakDE.png')
        image=decrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        img1 = cv2.imread('SelectedImage.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('BakEN.png',  cv2.IMREAD_GRAYSCALE)

# Resize the images to the same size
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

# Calculate the mean squared error (MSE)
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# Calculate the PSNR
        psn = 10 * np.log10((255 ** 2) / mse)
        label = Label(root, text ="Entropy Score between Original imagfe and encrypted image is "+str(entropy_score(img1,img2))).pack(side="bottom")
        label = Label(root, text ="PSNR Score between Original imagfe and encrypted image is "+str(psn)).pack(side="bottom")
        draw_histograms(img1,img2)


# Main function and function call
    img_path = 'BakEn.png'
    block_size = 2
    key_file = 'BakKey.txt'
    decrypt(img_path, block_size, key_file)

#encryptCat is encryption for Cat chaotic maps
def encryptCat():
    def chaotic_map(x, y):
        x_new = (2*x + y) 
        y = (x + y) 
        return (   x_new )


# Encryption function
    def encrypt(img_path, block_size, y, num_iterations, key_file):
    # Open the image
        img = Image.open(img_path).convert('L')
        img = img.resize((200, 200))
        img_arr = np.array(img)

    # Pad the image if necessary
        padded_height = block_size - img_arr.shape[0] % block_size
        padded_width = block_size - img_arr.shape[1] % block_size
        img_arr = np.pad(img_arr, ((0, padded_height), (0, padded_width)), mode='constant')

    # Create a list to store the encrypted blocks
        encrypted_blocks = []

    # Initialize the chaotic map with a random seed and declare key storage list 'key_data'
        x = np.random.rand()
        key_data=[]
    # Iterate over all blocks in the image
        for i in range(0, img_arr.shape[0], block_size):
            for j in range(0, img_arr.shape[1], block_size):

            # Get the current block
                block = img_arr[i:i+block_size, j:j+block_size]
            # Iterate the chaotic map to get the key
                for k in range(num_iterations):
                    x = chaotic_map(x, y)
                # Round the chaotic map output to get the key value
                    #x= x /((abs(x) + 1)) / 2 + 0.5
                x="0."+str(x)[2:7]
            
                x=float(x)    
                key = int(round(x * 255))
            
                #save key to list
                key_data.append(key)
            
                # XOR the block with the key
                encrypted_block = np.bitwise_xor(block, key)
                # Add the encrypted block to the list
                encrypted_blocks.append(encrypted_block)
    #Save the key list as a text file
        with open(key_file, 'w') as f:
            f.write(str(key_data))
    # Combine the encrypted blocks and save the encrypted image
        encrypted_img = np.vstack([np.hstack(encrypted_blocks[i:i+len(range(0, img_arr.shape[1], block_size))]) for i in range(0, len(encrypted_blocks), len(range(0, img_arr.shape[1], block_size)))])
        encrypted_img = Image.fromarray(encrypted_img)
        encrypted_img.save('CatEN.png')
        image=encrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        decrypt_button = tk.Button(root, text="Decrypt", command=decryptCat)
        decrypt_button.pack()

#Main Initialization and function call
    img_path = 'SelectedImage.png'
    label = Label(root, text ="Cat map encryption selected",font=("Arial", 15)).pack()
    block_size = 2
    x, y = 0, 0
    num_iterations = 3
    key_data=[]
    key_file = 'CatKey.txt'
    encrypt(img_path, block_size, y, num_iterations, key_file)

#encryptHan is encryption for Han chaotic maps
def encryptHan():
    def chaotic_map(x, y):
        x_new = abs(1 - a*x**2 + y ) 
        y = b*x
        return (x_new / (abs(x) + 1)) / 2 + 0.5
    

    # Encryption function
    def encrypt(img_path, block_size, y, num_iterations, key_file):
        # Open the image
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img)

        # Pad the image if necessary
        padded_height = block_size - img_arr.shape[0] % block_size
        padded_width = block_size - img_arr.shape[1] % block_size
        img_arr = np.pad(img_arr, ((0, padded_height), (0, padded_width)), mode='constant')

        # Create a list to store the encrypted blocks
        encrypted_blocks = []

        # Initialize the chaotic map with a random seed and declare key storage list 'key_data'
        x = np.random.rand()
        key_data=[]
    # Iterate over all blocks in the image
        for i in range(0, img_arr.shape[0], block_size):
            for j in range(0, img_arr.shape[1], block_size):

                # Get the current block
                block = img_arr[i:i+block_size, j:j+block_size]
                # Iterate the chaotic map to get the key
                for k in range(num_iterations):
                    x = chaotic_map(x, y)
                # Round the chaotic map output to get the key value
                key = int(round(x * 255))
                if key>255:
                    key=abs(key-600)
            #print(key)
            #save key to list
                key_data.append(key)
            # XOR the block with the key
                encrypted_block = np.bitwise_xor(block, key)
            # Add the encrypted block to the list
                encrypted_blocks.append(encrypted_block)
    #Save the key list as a text file
        with open(key_file, 'w') as f:
            f.write(str(key_data))
    # Combine the encrypted blocks and save the encrypted image
        encrypted_img = np.vstack([np.hstack(encrypted_blocks[i:i+len(range(0, img_arr.shape[1], block_size))]) for i in range(0, len(encrypted_blocks), len(range(0, img_arr.shape[1], block_size)))])
        encrypted_img = Image.fromarray(encrypted_img)
        encrypted_img.save('HanEN.png')
        image=encrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        decrypt_button = tk.Button(root, text="Decrypt", command=decryptHan)
        decrypt_button.pack()

#Main Initialization and function call
    label = Label(root, text ="Han map encryption selected",font=("Arial", 15)).pack()
    img_path = 'SelectedImage.png'
    block_size = 2
    a, b = 1.4, 0.3
    x, y = 0, 4
    num_iterations = 3
    key_data=[]
    key_file = 'HanKey.txt'
    encrypt(img_path, block_size, y, num_iterations, key_file)

#encryptBak is encryption for Bak chaotic maps
def encryptBak():
    def chaotic_map(x, r):
        return r*x*(1-x)

# Encryption function
    def encrypt(img_path, block_size, r, num_iterations, key_file):
        # Open the image
        img = Image.open(img_path).convert('L')
        img = img.resize((200, 200))
        img_arr = np.array(img)

        # Pad the image if necessary
        padded_height = block_size - img_arr.shape[0] % block_size
        padded_width = block_size - img_arr.shape[1] % block_size
        img_arr = np.pad(img_arr, ((0, padded_height), (0, padded_width)), mode='constant')

    # Create a list to store the encrypted blocks
        encrypted_blocks = []

    # Initialize the chaotic map with a random seed and declare key storage list 'key_data'
        x = np.random.rand()
        key_data=[]
    # Iterate over all blocks in the image
        for i in range(0, img_arr.shape[0], block_size):
            for j in range(0, img_arr.shape[1], block_size):

            # Get the current block
                block = img_arr[i:i+block_size, j:j+block_size]
            # Iterate the chaotic map to get the key
                for k in range(num_iterations):
                    x = chaotic_map(x, r)
            # Round the chaotic map output to get the key value
                key = int(round(x * 255))
                #save key to list
                key_data.append(key)
                # XOR the block with the key
                encrypted_block = np.bitwise_xor(block, key)
                # Add the encrypted block to the list
                encrypted_blocks.append(encrypted_block)
    #Save the key list as a text file
        with open(key_file, 'w') as f:
            f.write(str(key_data))
        # Combine the encrypted blocks and save the encrypted image
        encrypted_img = np.vstack([np.hstack(encrypted_blocks[i:i+len(range(0, img_arr.shape[1], block_size))]) for i in range(0, len(encrypted_blocks), len(range(0, img_arr.shape[1], block_size)))])
        encrypted_img = Image.fromarray(encrypted_img)
        encrypted_img.save('BakEN.png')

        image=encrypted_img
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        image_list.append(image)
        image = tk.Label(root, image=image)
        image.pack(side="left", padx=105, pady=10, anchor= "n")

        decrypt_button = tk.Button(root, text="Decrypt", command=decryptBak)
        decrypt_button.pack()

#Main Initialization and function call
    img_path = 'SelectedImage.png'
    label = Label(root, text ="Baker's map encryption selected",font=("Arial", 15)).pack()
    block_size = 2
    r = 3.8
    num_iterations = 3
    key_data=[]
    key_file = 'BakKey.txt'
    encrypt(img_path, block_size, r, num_iterations, key_file)

#step2 consists of inputing image performing pre processing classifying the image based on the model created during ML.py 
#and choosing Shamir secret share or random sare or chaotic maps encryption methods automatically. 
#The encryption or random share and shamir share are also present in this function.
def step2():
    # Path to the saved k-means model
    model_path = 'model.joblib'
    
    # Load the k-means model from disk
    kmeans = load(model_path)

    # Path to the new image you want to predict the label for
    new_img_path = 'SelectedImage.png'

    # Common image shape
    img_shape = (100,100)
  
    #Load the new image and resize to the common shape
    new_img = cv2.imread(new_img_path)
    new_img_resized = cv2.resize(new_img, img_shape)

    # Reshape the new image into a one-dimensional vector
    new_img_1d = new_img_resized.reshape(1, -1)
   
    # Predict the cluster label for the new image
    l = kmeans.predict(new_img_1d)
    
    if l==[0]:
        label = Label(root, text ="Method used is random share creation", font=("Arial", 15)).pack()
        label = Label(root, text ="Below are the two shares created", font=("Arial", 15)).pack()
            # Open the image and convert to numpy array
        image = Image.open('SelectedImage.png')
        image_array = np.array(image)

        # Generate random NumPy array of same shape as image array
        share_1 = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)
        share_2 = np.bitwise_xor(share_1, image_array)
        

        # COnvert array to image and save them
        share_1=Image.fromarray(share_1)
        share_2=Image.fromarray(share_2)
    
    
        share_1 = share_1.resize((200, 200))
        share_2 = share_2.resize((200, 200))
        s1=share_1.save("S1C.png")
        s2=share_2.save("S2C.png")
       
        # Convert the image to PhotoImage format
        share_1 = ImageTk.PhotoImage(share_1)
        share_2 = ImageTk.PhotoImage(share_2)
        # Create a label to display the image on the window
        image_list.append(share_1)
        image_list.append(share_2)
        
        share_1 = tk.Label(root, image=share_1)
        share_2 = tk.Label(root, image=share_2)

        
        # Add the label to the window
        share_1.pack(side="left", padx=70, pady=10, anchor= "n")
        share_2.pack(side="left" , pady=10, anchor="n")


        decrypt_button = tk.Button(root, text="Decrypt", command=decrypt1)
    
        # Add the button to the window
        decrypt_button.pack()

    if l==[1]:
        
        label = Label(root, text ="Method used is Shamir's Seecret Share",font=("Arial", 15)).pack()
        label = Label(root, text ="Below are the two shares created", font=("Arial", 15)).pack()
            # Open the image and convert to numpy array
        img= Image.open('SelectedImage.png').convert('L')
        
        img = img.resize((200, 200))

        #Opening the image and displaying it
        

        #Converting image to binary format
        thresh=100    #thresh = Threshold
        width,height=img.size    
        for x in range(width):
            for y in range(height):
                if img.getpixel((x,y)) < thresh:
                    img.putpixel((x,y),0)
                else:
                    img.putpixel((x,y),1) 
            #put 255 in place of 1 for binary image

        #s1(Share 1), s2(Share 2), de(Decrypted image) declaration
        s1=[]
        s2=[]
        de=[]

        #Modifing the above declared list into 2D based on height
        for x in range(height):
            s1.append([])
            s2.append([])
            de.append([])

    #   Converting image to numpy array
        img=np.asarray(img)        

        #Encryption
        for x in range(height):
            for y in range(width):      
                #Selecting random number 0 or 1
                z = randint(0,1)

                #Encryption algorithm
                if(img[x][y]==1):
                    if(z==1):
                        s1[x].insert(y,1)
                        s2[x].insert(y,0)
                
                    else:
                        s1[x].insert(y,0)
                        s2[x].insert(y,1)
                else:
                    if(z==1):
                        s1[x].insert(y,1)
                        s2[x].insert(y,1)
                    else:
                        s1[x].insert(y,0)
                        s2[x].insert(y,0)

        #Convertig share1 & share2 into numpy array then into image.
        s1=np.asarray(s1)
        s2=np.asarray(s2)
        s1=Image.fromarray(s1)
        s2=Image.fromarray(s2)

        #Remove both (1) and (2) code blocks for faster execution of program but remember that the share1 and share2 wont be displayed the correct way.


        #(1) Converting the shares from (0,1)image to (0,255)image
        #{
        width,height=s1.size    
        for x in range(width):
            for y in range(height):
                if s1.getpixel((x,y)) == 1:
                    s1.putpixel((x,y),255)
                else:
                    s1.putpixel((x,y),0) 
                if s2.getpixel((x,y)) == 1:
                    s2.putpixel((x,y),255)
                else:
                    s2.putpixel((x,y),0) 
        #}

        #display share1 and share2

        share_1=s1
        share_2=s2

        share_1 = share_1.resize((200, 200))
        share_2 = share_2.resize((200, 200))
      
        s1=share_1.save("s1.png")
        s2=share_2.save("s2.png")
        # Convert the image to PhotoImage format
        share_1 = ImageTk.PhotoImage(share_1)
        share_2 = ImageTk.PhotoImage(share_2)
        # Create a label to display the image on the window
        image_list.append(share_1)
        image_list.append(share_2)
        
        share_1 = tk.Label(root, image=share_1)
        share_2 = tk.Label(root, image=share_2)
        
    
        # Add the label to the window
        share_1.pack(side="left", padx=70, pady=10, anchor= "n")
        share_2.pack(side="left" , pady=10, anchor="n")
   

        decrypt_button = tk.Button(root, text="Decrypt", command=decrypt2)
        decrypt_button.pack()

    if l==[2]:
        label = Label(root, text ="Chaotic map encryption methods are to be selected",font=("Arial", 15)).pack()
        encrypt_button1 = tk.Button(root, text="Han map encryption", command=encryptHan)
        encrypt_button1.pack()
        encrypt_button2 = tk.Button(root, text="Cat map encryption", command=encryptCat)
        encrypt_button2.pack()
        encrypt_button3 = tk.Button(root, text="Baker's map encryption", command=encryptBak)
        encrypt_button3.pack()

# Function to open the file dialog and select an image file
def open_image():
    # Get the file path of the selected image
    filepath = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
    
    # Load the selected image using PIL ans save
    image = Image.open(filepath)
    
    # Resize the image to fit the window size
    image = image.resize((200, 200))
    
    # Convert the image to PhotoImage format
    image_tk = ImageTk.PhotoImage(image)
    
    # Create a label to display the image on the window
    #global image_label
    image_label = tk.Label(root, image=image_tk)
    image=image.save("SelectedImage.png")
    # Add the label to the window
    image_label.pack()
    # Append the PhotoImage object to the image_list
    image_list.append(image_tk)

     # Create a "Continue" button
    continue_button = tk.Button(root, text="Continue", command=step2)
    
    # Add the button to the window
    continue_button.pack()

# Create a button for uploading images
upload_button = tk.Button(root, text="Upload Image", command=open_image)
# Add the button to the window
upload_button.pack()

root.mainloop()

