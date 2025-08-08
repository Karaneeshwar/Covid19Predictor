# %%
import numpy as np
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# %%
def categorize(img):
    categories = np.digitize(img, bins=np.linspace(img.min(), img.max(), num=6))
    smoothed_categories = gaussian_filter(categories, sigma=0.2)
    print(smoothed_categories.min(), '\t', smoothed_categories.max())
    cat1 = np.digitize(smoothed_categories, bins=np.linspace(smoothed_categories.min(), smoothed_categories.max(), num=smoothed_categories.max()))
    return cat1

# %%
def reduceImg(i,j,categories):
    stack = [(i, j)]

    while stack:
        x, y = stack.pop()
        # Set the current cell to 1
        categories[x][y] = 0

        # Check the 4 neighboring cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # Ensure we stay within bounds and process only valid cells
            if 0 <= nx < categories.shape[0] and 0 <= ny < categories.shape[1]:
                if  categories[nx][ny] == 1:  # Only visit cells greater than 1
                    stack.append((nx, ny))
    return

# %%
def filter1(cat1):
    found = False
    cat2 = gaussian_filter(cat1, sigma=0.2)
    reduceImg(0,0,cat2)
    return cat2
        

# %%
def reduceImg2(i,j,categories):
    stack = [(i, j)]

    while stack:
        x, y = stack.pop()
        # Set the current cell to 1
        categories[x][y] = 0

        # Check the 4 neighboring cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # Ensure we stay within bounds and process only valid cells
            if 0 <= nx < categories.shape[0] and 0 <= ny < categories.shape[1]:
                if categories[nx][ny] >= 2:  # Only visit cells greater than 1
                    stack.append((nx, ny))
    return

# %%
def filter2(cat2):
    found = False
    cat3 = cat2.copy()
    for i in range(len(cat2)):
        for j in range(len(cat2[i])):
            if (cat3[i][j]>=2):
                reduceImg2(i,j,cat3)
                found = True
                break
        if found:
            break
    i = 0
    f=0
    b=0
    while i < len(cat3):
        if 1 in cat3[i]:
            if f==0:
                f=1
        else:
            if f==1:
                b = 1
        if b==1:
            cat3[i] = np.zeros(shape=cat3[i].shape)
        i=i+1
    return cat3

# %%
import numpy as np

def crop_to_mask(img: np.ndarray, mask: np.ndarray, pad: int = 0):
    """
    Crop `img` and `mask` to the minimal bounding box around mask>0 pixels,
    with an optional `pad` in pixels on each side.
    
    Parameters
    ----------
    img : np.ndarray
        The original image (H×W or H×W×C).
    mask : np.ndarray
        A binary mask of shape (H×W), zeros outside region of interest.
    pad : int
        How many extra pixels to include on each side of the box (default 0).
    
    Returns
    -------
    img_crop : np.ndarray
        Cropped version of `img`.
    mask_crop : np.ndarray
        Cropped version of `mask`.
    """
    # find all nonzero mask coords
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        # nothing to crop—return originals
        return img, mask

    # compute bounding box
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # apply padding, clamped to image edges
    x0 = max(x0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, mask.shape[1] - 1)
    y1 = min(y1 + pad, mask.shape[0] - 1)

    # slice out the ROI
    if img.ndim == 2:
        img_crop  = img[y0:y1+1, x0:x1+1]
    else:
        img_crop  = img[y0:y1+1, x0:x1+1, ...]
    mask_crop = mask[y0:y1+1, x0:x1+1]

    return img_crop, mask_crop


# %%
def preprocessLungCT(path):
    img = img_to_array(Image.open(path).convert('L'))
    cat1 = categorize(img)
    cat2 = filter1(cat1)
    mask = np.squeeze(filter2(cat2))
    img_crop, mask_crop = crop_to_mask(img, mask, pad=5)
    img_crop = np.squeeze(img_crop)
    pimg = img_crop*mask_crop
    return pimg

# %%
def categorize(img):
    categories = np.digitize(img, bins=np.linspace(img.min(), img.max(), num=6))
    from scipy.ndimage import gaussian_filter
    smoothed_categories = gaussian_filter(categories, sigma=0.2)
    cat1 = np.digitize(smoothed_categories, bins=np.linspace(smoothed_categories.min(), smoothed_categories.max(), num=4))
    return cat1

def reduceImg(i,j,categories):
    stack = [(i, j)]

    while stack:
        x, y = stack.pop()
        # Set the current cell to 1
        categories[x][y] = 0

        # Check the 4 neighboring cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # Ensure we stay within bounds and process only valid cells
            if 0 <= nx < categories.shape[0] and 0 <= ny < categories.shape[1]:
                if  categories[nx][ny] == 1:  # Only visit cells greater than 1
                    stack.append((nx, ny))
    return

def filter1(cat1):
    cat2 = gaussian_filter(cat1, sigma=0.2)
    for i in range(len(cat1)):
        for j in range(len(cat1[i])):
            if (cat1[i][j]==1):
                reduceImg(i,j,cat2)
                break
    return cat2

def reduceImg2(i,j,categories):
    stack = [(i, j)]

    while stack:
        x, y = stack.pop()
        # Set the current cell to 1
        categories[x][y] = 0

        # Check the 4 neighboring cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # Ensure we stay within bounds and process only valid cells
            if 0 <= nx < categories.shape[0] and 0 <= ny < categories.shape[1]:
                if categories[nx][ny] >= 2:  # Only visit cells greater than 1
                    stack.append((nx, ny))
    return

def filter2(cat2):
    cat3 = cat2.copy()
    for i in range(len(cat2)):
        for j in range(len(cat2[i])):
            if (cat3[i][j]>=2):
                reduceImg2(i,j,cat3)
                break
    i = 0
    f=0
    b=0
    while i < len(cat3):
        if 1 in cat3[i]:
            if f==0:
                f=1
        else:
            if f==1:
                b = 1
        if b==1:
            cat3[i] = np.zeros(shape=cat3[i].shape)
        i=i+1
    return cat3

def crop_to_mask(img: np.ndarray, mask: np.ndarray, pad: int = 0):
    """
    Crop `img` and `mask` to the minimal bounding box around mask>0 pixels,
    with an optional `pad` in pixels on each side.

    Parameters
    ----------
    img : np.ndarray
        The original image (H×W or H×W×C).
    mask : np.ndarray
        A binary mask of shape (H×W), zeros outside region of interest.
    pad : int
        How many extra pixels to include on each side of the box (default 0).

    Returns
    -------
    img_crop : np.ndarray
        Cropped version of `img`.
    mask_crop : np.ndarray
        Cropped version of `mask`.
    """
    # find all nonzero mask coords
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        # nothing to crop—return originals
        return img, mask

    # compute bounding box
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # apply padding, clamped to image edges
    x0 = max(x0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, mask.shape[1] - 1)
    y1 = min(y1 + pad, mask.shape[0] - 1)

    # slice out the ROI
    if img.ndim == 2:
        img_crop  = img[y0:y1+1, x0:x1+1]
    else:
        img_crop  = img[y0:y1+1, x0:x1+1, ...]
    mask_crop = mask[y0:y1+1, x0:x1+1]

    return img_crop, mask_crop


def preprocessLungCT(img):
    img = np.array(Image.open(img).convert('L'))
    print(img.shape)
    cat1 = categorize(img)
    cat2 = filter1(cat1)
    mask = filter2(cat2)
    img_crop, mask_crop = crop_to_mask(img, mask, pad=5)
    pimg = img_crop*mask_crop
    return pimg

# Load Pretrained Model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # Global average pooling

# Function to Extract Features
def extract_feature(image_path):
    pimg_array = preprocessLungCT(image_path)
    arr = pimg_array
    # ensure uint8
    if arr.dtype != np.uint8:
        # scale/clip floats or cast ints
        arr = np.clip(arr, 0, 255).astype('uint8')
    pil = Image.fromarray(arr, mode='L').convert('RGB')
    img_rgb = pil.resize((224,224), resample=Image.BILINEAR)
    img_array = img_to_array(img_rgb)
    img_array = preprocess_input(img_array)  # Preprocess for ResNet
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = base_model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten to 1D array


pimg = extract_feature(r'F:\Sem 6\DSA\Lab\Covid19Predictor\webapp\static\uploads\cov.png')
