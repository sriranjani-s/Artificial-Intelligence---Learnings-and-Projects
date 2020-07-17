import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import math
import os

# The output path to save all the figures/output images generated in Task 1
picFolderName = 'R00182510_Task1_Output_Images'
picPath = os.path.join(os.getcwd(),picFolderName)
if not os.path.exists(picPath):
    os.makedirs(picPath)


# Subtask B - creation of 12 gaussian smoothing kernels for the resized grey image.
def Task1_Gaussian_smoothing(img_resized):
    gaussian_img = []
    for k in range(12):
        sigma = 2 ** (k / 2)
        x, y = np.meshgrid(np.arange(-3 * sigma, 3 * sigma), np.arange(-3 * sigma, 3 * sigma))
        gaussian_kernel = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

        # save the kernel image to the output path as given in picPath
        fig = plt.figure()
        plt.imshow(gaussian_kernel)
        fig.savefig(picPath+'\gaussian_kernel'+str(k+1)+'.png')

        # apply the kernel to the image
        gaussian_img.append(cv2.filter2D(img_resized, -1, gaussian_kernel))

        # save the gaussian smoothed image to the output path as given in picPath
        fig = plt.figure()
        plt.imshow(gaussian_img[k])
        fig.savefig(picPath+'\gaussian_img' + str(k + 1) + '.png')

    # Function returns all the smoothed/scale space images as a list
    return gaussian_img


# Subtask C - Calculate DOG images at all scales.
def Task1_calculate_Difference_Of_Gaussians(gaussian_img):

    DOGs = [gaussian_img[i + 1] - gaussian_img[i] for i in range(len(gaussian_img) - 1)]

    # save all the DOG images to the output path specified in picPath
    for i, DOG in enumerate(DOGs):
        fig = plt.figure()
        plt.imshow(DOG)
        fig.savefig(picPath + '\DOG' + str(i + 1) + '.png')

    # Function returns a list of DOG images
    return DOGs

# Subtask D - supress non maxima in scale space to find key points with coordinate (x-point, y-point, scale)
# Key point is determined after matching its value with 26 immediate neighbours
# (8 coordinates at the current scale + 9 coordinates at one scale below + 9 coordinates at one scale above)
def Task1_non_maximum_suppression(DOGs,T):
    points = []

    # x represents the scale, y represents point-1 coordinate, z represents point-2 coordinate
    for x in range(1, len(DOGs)-1):
        for y in range(1,len(DOGs[x])-1):
            for z in range(1,len(DOGs[x][y])-1):
                if ((DOGs[x][y][z]>T) and
                    (DOGs[x][y][z]>DOGs[x][y-1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x][y-1][z]) and
                    (DOGs[x][y][z]>DOGs[x][y-1][z+1]) and
                    (DOGs[x][y][z]>DOGs[x][y][z-1]) and
                    (DOGs[x][y][z]>DOGs[x][y][z+1]) and
                    (DOGs[x][y][z]>DOGs[x][y+1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x][y+1][z]) and
                    (DOGs[x][y][z]>DOGs[x][y+1][z+1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y-1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y-1][z]) and
                    (DOGs[x][y][z]>DOGs[x-1][y-1][z+1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y][z-1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y][z]) and
                    (DOGs[x][y][z]>DOGs[x-1][y][z+1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y+1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x-1][y+1][z]) and
                    (DOGs[x][y][z]>DOGs[x-1][y+1][z+1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y-1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y-1][z]) and
                    (DOGs[x][y][z]>DOGs[x+1][y-1][z+1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y][z-1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y][z]) and
                    (DOGs[x][y][z]>DOGs[x+1][y][z+1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y+1][z-1]) and
                    (DOGs[x][y][z]>DOGs[x+1][y+1][z]) and
                    (DOGs[x][y][z]>DOGs[x+1][y+1][z+1])):
                    points.append((y,z,x))

    # Function returns all the keypoints in the format (x, y, sigma) as a list
    return points


# Subtask E - Calculation of the derivatives in scale space for the output images from Subtask B
def Task1_calculate_derivatives(gaussian_img):
    dx = np.array([[1, 0, -1]])
    dy = np.array([[1, 0, -1]])
    dy = dy.T

    img_x = []
    img_y = []
    for i, img in enumerate(gaussian_img):
        # Calculate gradient-X
        dx_x = cv2.filter2D(img, -1, dx)
        #Calcualte gradient-Y
        dy_y = cv2.filter2D(img, -1, dy)

        img_x.append(dx_x)
        img_y.append(dy_y)

        # save all the gradient-X images to the output path specified in picPath
        fig = plt.figure()
        plt.imshow(dx_x)
        fig.savefig(picPath + '\derivative_x_gauss' + str(i + 1) + '.png')

        # save all the gradient-Y images to the output path specified in picPath
        fig = plt.figure()
        plt.imshow(dy_y)
        fig.savefig(picPath + '\derivative_y_gauss' + str(i + 1) + '.png')

    # Function returns gradient-X and gradient-Y for each image from Subtask B
    return(img_x, img_y)


# Subtask F - calculating gradient length and direction around each keypoint
def Task1_orientation_histogram(img_x, img_y, points, bin_width):
    orientation = []
    for point in points:
        x = []
        y = []
        for k in range(-3, 4, 1):
            x.append(math.ceil(abs(point[0] + ((3 / 2) * k * point[2]))))
            y.append(math.ceil(abs(point[1] + ((3 / 2) * k * point[2]))))
        # create a 7x7 matrix of points q,r considering the keypoint and it neighbourhood
        q, r = np.meshgrid(x, y)
        # create an orientation histogram of zeroes of size 36. Each bin width in the histogram is 10 degrees
        hist = np.zeros(36, dtype=np.float32)

        for i in range(len(q)):
            for j in range(len(q[1])):
                # locate the gradient images for the required scale(sigma)
                sigma_dx = img_x[point[2]]
                sigma_dy = img_y[point[2]]
                # convert floating point values to integer coordinates in order to look up in the gradient images
                pt1 = abs(int(q[i][j]))
                pt2 = abs(int(r[i][j]))
                # exclude boundary conditions
                if (pt1 >= len(sigma_dx)) or (pt2 >= len(sigma_dx[i])):
                    continue
                else:
                    # Retrieve the Gx and Gy value at the coordinates (q,r)
                    Gx = sigma_dx[pt1][pt2]
                    Gy = sigma_dy[pt1][pt2]
                    # calculate the magnitude
                    m_qr = np.sqrt(Gx ** 2 + Gy ** 2)
                    # calculate theta to determine the direction
                    theta_qr = (np.arctan2(Gy, Gx) + np.pi) * 180 / np.pi
                    # calculate the gaussian weighting function for each (q,r) point
                    w_qr = np.exp(-(q[i][j] ** 2 + r[i][j] ** 2) / ((9 * (point[2] ** 2)) / 2)) / ((9 * np.pi * (point[2] ** 2)) / 2)
                    weight = m_qr*w_qr
                    # assign bin numbers based on the theta value.
                    bin_no = int(theta_qr / bin_width)
                    # accumulate the weighted magnitude in the particular bin
                    hist[bin_no - 1] += weight
        # The bin with maximum value of the histogram corresponds to the orientation at the keypoint
        orientation.append(np.argmax(hist + 1) * 10)

    # Function returns the orientation/ direction of each keypoint
    return orientation

# Subtask G - Draw keypoints as circles and the direction as a line covering the radius of the circle
def Task1_draw_keypoints(points, input_image, orientation):
    for i, point in enumerate(points):
        color = (255,0,0)
        center = (int(point[1]/2), int(point[0]/2))
        size = int((3 * point[2])/2)
        cv2.circle(input_image, center, size, color, 1, cv2.LINE_AA)

        angle = (orientation[i]) * np.pi / 180
        orient = (int(round(np.cos(angle) * size)), int(round(np.sin(angle) * size)))
        cv2.line(input_image, center, (center[0] + orient[0], center[1] + orient[1]), color, 1, cv2.LINE_AA)

    return input_image


# Exexcute all subtasks in Task1 as functions
def Task1():
    input_image = cv2.imread("Assignment_MV_01_image_1.jpg")

    # Subtask A - convert the image to gray scale
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("gray", img)

    # convert the image values to float
    img = img.astype('float')

    height, width = img.shape[:2]
    # print(img.shape)

    # resize the image to double its size
    img_resized = cv2.resize(img, (2*width, 2*height))
    # print(img_resized.shape)

    # Subtask B - Apply gaussian smoothing
    gaussian_img = Task1_Gaussian_smoothing(img_resized)

    # Subtask C - Create Difference of gaussians
    DOGs = Task1_calculate_Difference_Of_Gaussians(gaussian_img)

    # Subtask D - supress non-maxima in scale space
    T = 10
    points = Task1_non_maximum_suppression(DOGs, T)
    print("No. of Keypoints generated: ", len(points))
    print(points)

    # Subtask E - Calculate gaussian derivatives
    img_x, img_y = Task1_calculate_derivatives(gaussian_img)

    # Subtask F - Calculate magnitude and direction at each keypoint
    bin_width = 10
    orientation = Task1_orientation_histogram(img_x, img_y, points, bin_width)

    # Draw the keypoints in the original image
    Image_with_keypoints = Task1_draw_keypoints(points, input_image, orientation)

    # Display final image with keypoints indicating te magnitude and direction
    cv2.imshow("result", Image_with_keypoints / 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.close('all')


# Exexcute all subtasks in Task2
def Task2():

    # Subtask A - Read input images
    input_image1 = cv2.imread("Assignment_MV_01_image_1.jpg")
    input_image2 = cv2.imread("Assignment_MV_01_image_2.jpg")

    # convert the imges to gray scale
    img1 = cv2.cvtColor(input_image1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(input_image2, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("gray_1", img1)
    # cv2.imshow("gray_2", img2)

    img1 = img1.astype('float')
    img2 = img2.astype('float')

    # Subtask B - Draw a rectangle around the interested area and crop the boxed image
    cv2.rectangle(img1, (360, 210), (430, 300), color=(255, 0, 0), thickness=3)
    cv2.imshow('Image-1 boxed to crop', img1 / 255)

    cropped_img1 = img1[210:300, 360:430]
    cv2.imshow("Image-1 Cropped Image to match", cropped_img1 / 255)

    # Subtask C - calculate mean and standard deviation of the cropped image and match it with Image 2 to
    # determine the matched area
    cropped_img1_mean = cropped_img1.mean()
    cropped_img1_std = cropped_img1.std()

    # determine the dimensions of the cropped image
    diff_height = 300 - 210
    diff_width = 430 - 360

    # print(diff_height, diff_width)

    cropped_img2 = []
    cropped_img2_mean = []
    cropped_img2_std = []
    cropped_img2_coord = []

    # Determine mean and standard deviation of the image patch at each pixel of image 2 with the same dimensions
    # as the cropped image
    for i in range(img2.shape[0] - (diff_height - 1)):
        for j in range(img2.shape[1] - (diff_width - 1)):
            cropped = img2[i:i + diff_height, j:j + diff_width]
            cropped_img2_mean.append(cropped.mean())
            cropped_img2_std.append(cropped.std())
            cropped_img2.append(cropped)
            cropped_img2_coord.append([i, j])

    image_correlation = []

    num_pixels = cropped_img1.shape[0] * cropped_img1.shape[1]

    # Determine the correlation between all the image patches from Image 2 and the original cropped image
    for i in range(len(cropped_img2)):
        covariance = np.sum((cropped_img1 - cropped_img1_mean) * (cropped_img2[i] - cropped_img2_mean[i])) / num_pixels
        correlation = covariance / (cropped_img1_std * cropped_img2_std[i])
        image_correlation.append(correlation)

    plt.plot(image_correlation)
    plt.title('Cross correlation at all Image-2 pixels')
    plt.show()

    # select the image patch with maximum correlation value
    max_corr = np.argmax(image_correlation)
    matched_y, matched_x = cropped_img2_coord[max_corr]

    # Draw a rectangle around the matched image patch
    cv2.rectangle(img2, (matched_x, matched_y), (matched_x + 70, matched_y + 90), color=(255, 0, 0), thickness=3)
    cv2.imshow("Image-2 matched image patch", img2 / 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Execute Task_1 or Task_2 by specifying the exact task name as a value to the execute variable
    execute = 'Task_1'

    if execute == 'Task_1':
        print('Executing Task-1')
        Task1()
    elif execute == 'Task_2':
        print('Executing Task-2')
        Task2()
    else:
        print("Task to execute is invalid")

    print('Execution completed!')

main()