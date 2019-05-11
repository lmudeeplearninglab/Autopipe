import cv2
import numpy as np
import matplotlib.pyplot as plt

#commented out to run locally
#from autopipe import serialize as serial

#added to run locally
import serialize as serial

# modify this to fit the size of images that are used for
# training and classifying
#images defined as column, row - due to cv2.resize call
END_IMAGE_SIZE = (200, 150)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def color_edges(img):
    """
    If the image is identified as a defect, then the edges will be colored
    and returned as an img
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 7
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    rho = 2
    theta = np.pi / 180
    threshold = 15

    # default 40 min 20 max
    min_line_len = 40
    max_line_gap = 15
    hough_img = hough_lines(edges, rho, theta, threshold, min_line_len,
                            max_line_gap)

    lower_left = (50, img.shape[0]-125)
    upper_left = (50, 180)
    upper_right = (img.shape[1]-50, 180)
    lower_right = (img.shape[1]-50, img.shape[0]-125)
    vertices = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    area_extracted_img = region_of_interest(hough_img, vertices)
    colored_img = weighted_img(area_extracted_img, img)
    return colored_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def draw_lines(img, lines, color=None, thickness=15):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if not color:
        color = [0, 0, 255]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines)

    return line_img


def resize(image, size):
    return cv2.resize(image, size)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def crop_image(image, gray=False):
    """Crop an image from range[rows, cols]"""
    #return image[180:365, 20:620, :] if not gray else image[180:365, 20:620]
	#utilze whole image instead of cropped image
	#cropping out bottom third of image
    rows, cols, depth = image.shape
    upper_row = int(0)
    lower_row = int(rows * 2 / 3)
    left_col = int(0)
    right_col = int(cols)
    return image[upper_row:lower_row, left_col:right_col, :] if not gray else image[upper_row:lower_row, left_col:right_col]

def flip_image(image):
    #returns vertically flipped image
    #return cv2.flip(image, 1)
    #returns horizontally flipped image
    return cv2.flip(image, 0)


def normalize(data):
    return (data / 255.0) - 0.5


def grad(img, *, orient='x', sobel_kernel=3):
    """Compute the gradient across an axis"""
    if orient == 'x':
        orientation = (1, 0)
    elif orient == 'y':
        orientation = (0, 1)
    else:
        raise ValueError('{} is not a correct orientation'.format(orient))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_64F, *orientation, ksize=sobel_kernel)
    return grad


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Takes the absolute value, scales the gradient and applies a threshold"""
    sobel = grad(img, orient=orient, sobel_kernel=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return binary_out(scaled_sobel, thresh)


def binary_out(data, thresh=(0, 255)):
    """Returns the binary thresholded image for a given threshold range"""
    binary_out = np.zeros_like(data)
    binary_out[(thresh[0] <= data) & (data <= thresh[1])] = 1

    return binary_out


def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """Returns the magnitude of the gradient in both x and y over a threshold"""
    sobel_x = grad(img, orient='x', sobel_kernel=sobel_kernel)
    sobel_y = grad(img, orient='y', sobel_kernel=sobel_kernel)

    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_mag_scaled = np.int8(grad_mag * 255 / (np.max(grad_mag)))
    return binary_out(grad_mag_scaled, mag_thresh)


def direction_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """Returns edges in a specific orientation"""
    sobel_x = grad(img, orient='x', sobel_kernel=sobel_kernel)
    sobel_y = grad(img, orient='y', sobel_kernel=sobel_kernel)

    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    return binary_out(direction, thresh)


def display_image(image, title, gray=False):
    plt.title(title)
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def canny(img, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, thresh[0], thresh[1])


def combine_thresholds(image, display=False):
    abs_binary = abs_sobel_thresh(image, orient='x', thresh=(50, 60))
    mag_binary = mag_thresh(image, sobel_kernel=5, mag_thresh=(80, 130))

    combined_binary = np.zeros_like(mag_binary)
    combined_binary[(mag_binary== 1) | (abs_binary == 1)] = 1

    if display:
        display_image(image, "undistorted image")
        display_image(abs_binary, "x gradient threshold", gray=True)
        display_image(mag_binary, "mag binary", gray=True)

    return combined_binary


def preprocess(data, real_time=False):
    """Subtracting the mean and dividing by 
    standard deviation of of the features
    """
    if real_time:
        data = np.array([serial.process_before_serialize(image) for image in data])
    else:
        data = np.array([resize(image, END_IMAGE_SIZE) for image in data])

    data = data.astype('float32')
    data = normalize(data)
    data = np.array([grayscale(image) for image in data])

    return data[:, :, :, np.newaxis]


if __name__ == "__main__":
    pass
