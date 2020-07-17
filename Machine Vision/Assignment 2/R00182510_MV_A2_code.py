import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(1234)


def pre_processing():
    # Task 1 - Part A
    chess_rows = 5
    chess_cols = 7
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = np.zeros((chess_rows * chess_cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)

    obj_points_array = []
    img_points_array = []

    images = glob.glob('./Assignment_MV_02_calibration/*.png')

    for img_name in images:
        img = cv2.imread(img_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return_val, corners = cv2.findChessboardCorners(gray_img, (chess_rows, chess_cols))

        # Extract and display the checkerboard corners to subpixel accuracy in all images
        if return_val == True:
            corners_refined = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), termination_criteria)
            cv2.drawChessboardCorners(img, (chess_rows, chess_cols), corners, return_val)

            obj_points_array.append(object_points)
            img_points_array.append(corners_refined)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    height, width = img.shape[:2]

    # Task 1 - Part B
    # Camera calibrationmatrix 𝑲u
    return_val, cam_matrix, dist_coeff, rot_vect, trans_vect = cv2.calibrateCamera(obj_points_array,
                                                                                   img_points_array, (width, height),
                                                                                   None, None)

    print("Camera calibration matrix: ")
    print(cam_matrix)

    return cam_matrix


def get_tracks(video_file):
    # Task 1 - Part C
    camera = cv2.VideoCapture(video_file)

    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # good features to track
    while camera.isOpened():
        ret_val, img = camera.read()
        if ret_val:
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            p0 = cv2.goodFeaturesToTrack(new_img, 500, 0.3, 7)
            break
    # refine feature point to subpixel accuracy
    p0_refined = cv2.cornerSubPix(new_img, p0, (11, 11), (-1, -1), termination_criteria)

    # Task 1 - Part D
    # initialise tracks
    index = np.arange(len(p0_refined))
    tracks = {}
    for i in range(len(p0_refined)):
        tracks[index[i]] = {0: p0_refined[i]}

    frame = 0
    while camera.isOpened():
        ret_val, img = camera.read()

        if not ret_val:
            break

        frame += 1

        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # calculate optical flow
        if len(p0_refined) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0_refined, None)

            # visualise points
            for i in range(len(st)):
                if st[i]:
                    cv2.circle(img, (p1[i, 0, 0], p1[i, 0, 1]), 2, (0, 0, 255), 2)
                    cv2.line(img, (p0_refined[i, 0, 0], p0_refined[i, 0, 1]), (
                    int(p0_refined[i, 0, 0] + (p1[i][0, 0] - p0_refined[i, 0, 0]) * 5),
                    int(p0_refined[i, 0, 1] + (p1[i][0, 1] - p0_refined[i, 0, 1]) * 5)), (0, 0, 255), 2)

            p0_refined = p1[st == 1].reshape(-1, 1, 2)
            index = index[st.flatten() == 1]

        # refresh features, if too many lost
        if len(p0_refined) < 250:
            new_p0 = cv2.goodFeaturesToTrack(new_img, 500 - len(p0_refined), 0.3, 7)
            # refine the feature point coordinates to sub-pixel accuracy
            new_p0_refined = cv2.cornerSubPix(new_img, new_p0, (11, 11), (-1, -1), termination_criteria)
            for i in range(len(new_p0_refined)):
                if np.min(np.linalg.norm((p0_refined - new_p0_refined[i]).reshape(len(p0_refined), 2), axis=1)) > 10:
                    p0_refined = np.append(p0_refined, new_p0_refined[i].reshape(-1, 1, 2), axis=0)
                    index = np.append(index, np.max(index) + 1)

        # update tracks
        for i in range(len(p0_refined)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0_refined[i]
            else:
                tracks[index[i]] = {frame: p0_refined[i]}

        # visualise last frames of active tracks
        for i in range(len(index)):
            for f in range(frame - 20, frame):
                if (f in tracks[index[i]]) and (f + 1 in tracks[index[i]]):
                    cv2.line(img,
                             (tracks[index[i]][f][0, 0], tracks[index[i]][f][0, 1]),
                             (tracks[index[i]][f + 1][0, 0], tracks[index[i]][f + 1][0, 1]),
                             (0, 255, 0), 1)

        k = cv2.waitKey(10)

        cv2.imshow("camera", img)

    camera.release()

    cv2.destroyAllWindows()

    return tracks, frame


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def calculate_correspondences(img, tracks, frame1, frame2):
    # Task 2 - Part A
    # calculate correspondences between 1st and last frame
    correspondences = []
    for track in tracks:
        if (frame1 in tracks[track]) and (frame2 in tracks[track]):
            x1 = [tracks[track][frame1][0, 1], tracks[track][frame1][0, 0], 1]
            x2 = [tracks[track][frame2][0, 1], tracks[track][frame2][0, 0], 1]
            correspondences.append((np.array(x1), np.array(x2)))

    for i in range(len(correspondences)):
        cv2.circle(img, (int(correspondences[i][1][1]), int(correspondences[i][1][0])), 2, (255, 0, 0), 2)
        cv2.circle(img, (int(correspondences[i][0][1]), int(correspondences[i][0][0])), 2, (255, 0, 0), 2)
        cv2.line(img, (int(correspondences[i][1][1]), int(correspondences[i][1][0])),
                 (int(correspondences[i][0][1]), int(correspondences[i][0][0])), (0, 255, 0), 2)

    cv2.imshow("camera", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return correspondences


def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame > last_frame:
            break
    return result


def calculate_fundamental_matrix(correspondences):
    N = len(correspondences)

    # Task 2 - Part B
    # Calculate the mean feature coordinates in the first and the last frame
    mu1 = np.mean([x1 for x1, x2 in correspondences], axis=0)
    mu2 = np.mean([x2 for x1, x2 in correspondences], axis=0)

    # Calculate standard deviations
    sum_sigma1 = 0
    sum_sigma2 = 0

    for x1, x2 in correspondences:
        sum_sigma1 += np.square(x1 - mu1)
        sum_sigma2 += np.square(x2 - mu2)

    sigma1 = np.array(np.sqrt(sum_sigma1 / N))
    sigma2 = np.array(np.sqrt(sum_sigma2 / N))

    T = np.array([[1 / sigma1[1], 0, -mu1[1] / sigma1[1]],
                  [0, 1 / sigma1[0], -mu1[0] / sigma1[0]],
                  [0, 0, 1]])

    T_prime = np.array([[1 / sigma2[1], 0, -mu2[1] / sigma2[1]],
                        [0, 1 / sigma2[0], -mu2[0] / sigma2[0]],
                        [0, 0, 1]])

    # Normalise all feature coordinates by translating and scaling it
    transformed_coord = []
    for x1, x2 in correspondences:
        y1 = np.matmul(T, x1)
        y2 = np.matmul(T_prime, x2)

        transformed_coord.append((np.array(y1), np.array(y2)))

    best_outliers = len(transformed_coord) + 1
    best_error = 1e100
    best_F = np.eye(3)
    for iteration in range(10000):
        A = np.empty((0, 9))
        # Task 2 - Part C
        # Select 8 feature correspondences at random
        samples_in = set(random.sample(range(len(transformed_coord)), 8))
        samples_out = set(range(len(transformed_coord))).difference(samples_in)

        # Matrix to calculate the fundamental matrix using the 8-point DLT algorithm
        for i in samples_in:
            y1, y2 = transformed_coord[i]
            Ai = np.kron(y1.T, y2.T)
            Ai = Ai.reshape(1, 9)
            #         Ai = Ai / np.sqrt(np.sum(Ai.flatten()**2))
            A = np.append(A, Ai, axis=0)

        # Task2 - Part D
        # Use the 8-point DLT algorithm to calculate the fundamental matrix 𝑭̂
        U, S, V = np.linalg.svd(A)
        F_hat = V[8, :].reshape(3, 3).T
        U, S, V = np.linalg.svd(F_hat)
        F_hat = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))

        # Apply the normalisation homographies to obtain fundamental matrix 𝑭
        F = np.matmul(T_prime.T, np.matmul(F_hat, T))

        # Task 2 - Part E
        Cxx = [[1, 0, 0],
               [0, 1, 1],
               [0, 0, 0]]

        count_outliers = 0
        accumulate_error = 0
        inliers = []
        outliers = []
        for i in samples_out:
            x1, x2 = correspondences[i]
            # calculate the value of the model equation
            gi = np.matmul(x2.T, np.matmul(F, x1))

            Vi1 = np.matmul(F, x1)
            Vi2 = np.matmul(F.T, x2)
            #  calculate the variance of the model equation
            variance_i = np.matmul(x2.T, np.matmul(F, np.matmul(Cxx, Vi2))) + np.matmul(x1.T, np.matmul(F.T,
                                                                                                        np.matmul(Cxx,
                                                                                                                  Vi1)))
            # Task2 - Part F
            # Determine for each of these correspondences if they are an outlier
            test_stat_i = np.square(gi) / np.square(variance_i)

            if test_stat_i > 6.635:
                count_outliers += 1
                outliers.append(correspondences[i])
            else:
                accumulate_error += test_stat_i
                inliers.append(correspondences[i])
        # Task 2 - Part G
        # Remove all outliers for the selection of eight points which yieldedthe least number of outliers
        if count_outliers < best_outliers:
            best_error = accumulate_error
            best_outliers = count_outliers
            best_F = F
            best_inlier_coord = inliers
            best_outlier_coord = outliers
        elif count_outliers == best_outliers:
            if accumulate_error < best_error:
                best_error = accumulate_error
                best_outliers = count_outliers
                best_F = F
                best_inlier_coord = inliers
                best_outlier_coord = outliers

    return best_F, best_outlier_coord, best_inlier_coord


def calculate_epipoles(F):
    # calculate epipole coordinates
    U, S, V = np.linalg.svd(F)
    e1 = V[2, :]
    U, S, V = np.linalg.svd(F.T)
    e2 = V[2, :]
    return e1, e2


def main():
    # Task 1 - part A and part B
    K = pre_processing()

    # Task 1 - part C and part D
    tracks, frames = get_tracks("Assignment_MV_02_video.mp4")

    # Task 2 - part A
    f1 = 0
    f2 = frames

    images = extract_frames("Assignment_MV_02_video.mp4", [f1, f2])
    # width = images[f1].shape[1]
    # height = images[f1].shape[0]

    copy_img = images[f2].copy()
    correspondences = calculate_correspondences(copy_img, tracks, 0, int(frames))

    # Task 2 - part B, part C, part D, part E , part F and part G
    F, outliers, inliers = calculate_fundamental_matrix(correspondences)

    # Task 2 - part H
    img = images[f2].copy()
    for i in range(len(inliers)):
        cv2.circle(img, (int(inliers[i][1][1]), int(inliers[i][1][0])), 2, (255, 0, 0), 2)
        cv2.circle(img, (int(inliers[i][0][1]), int(inliers[i][0][0])), 2, (255, 0, 0), 2)
        cv2.line(img, (int(inliers[i][1][1]), int(inliers[i][1][0])), (int(inliers[i][0][1]), int(inliers[i][0][0])),
                 (0, 255, 0), 2)

    for i in range(len(outliers)):
        cv2.circle(img, (int(outliers[i][1][1]), int(outliers[i][1][0])), 2, (255, 0, 0), 2)
        cv2.circle(img, (int(outliers[i][0][1]), int(outliers[i][0][0])), 2, (255, 0, 0), 2)
        cv2.line(img, (int(outliers[i][1][1]), int(outliers[i][1][0])),
                 (int(outliers[i][0][1]), int(outliers[i][0][0])), (0, 0, 255), 2)

    # display the inliers (in green) and outliers (in red)
    cv2.imshow("outliers_red and inliers_green", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\n Fundamental Matrix')
    print(F)

    # calculate epipole coordinates
    e1, e2 = calculate_epipoles(F)

    print('\n Epipoles')
    print(e1 / e1[2])
    print(e2 / e2[2])
    # 0le(images[f2], (int(e2[0] / e2[2]), int(e2[1] / e2[2])), 3, (0, 0, 255), 2)

    # cv2.imshow("epipole_img1", images[f1])
    # cv2.imshow("epipole _img2", images[f2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Task 3 - Part A
    # calculate essential matrix
    E_hat = np.matmul(K.T, np.matmul(F, K))

    # To ensure non-zero singular values of 𝑬 are identical
    U, S, V = np.linalg.svd(E_hat)
    S = (S[0] + S[1]) / 2
    E = np.matmul(U, np.matmul(np.diag([S, S, 0]), V))

    print('\n Essential matrtix')
    print(E)

    # Rotation matrices of the singular value decomposition have positive determinants
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(V) < 0:
        V[2, :] *= -1

    # Task 3 - Part B
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    # calculate baseline beta
    time = int(frames) / 30
    speed = 50 / 3.6
    beta = time * speed

    # four potential combinations of rotation matrices 𝑹 and
    # translation vector 𝒕 between the first and the last frame
    R1_T = np.matmul(U, np.matmul(W, V.T))
    R2_T = np.matmul(U, np.matmul(W.T, V.T))

    t1_temp = np.matmul(U, np.matmul(Z, U.T))
    t2_temp = np.matmul(U, np.matmul(Z, U.T))

    skew_matrix1 = beta * t1_temp
    skew_matrix2 = -beta * t2_temp

    R_T_t_pos = np.array([skew_matrix1[2, 1], skew_matrix1[0, 2], skew_matrix1[1, 0]])
    R_T_t_neg = np.array([skew_matrix2[2, 1], skew_matrix2[0, 2], skew_matrix2[1, 0]])

    t1 = np.matmul(np.linalg.inv(R1_T), R_T_t_pos)
    t2 = np.matmul(np.linalg.inv(R1_T), R_T_t_neg)

    # Task 3 - Part C
    all_3D_points = []
    scene_points_count = []
    soln = []
    for Ri in (R1_T, R2_T):
        for ti in (t1, t2):
            coord_3D_points = []
            best_scene_points = 0
            for x1, x2 in inliers:
                # calculate directions 𝒎 and 𝒎′
                m1 = np.matmul(np.linalg.inv(K), x1)
                m2 = np.matmul(np.linalg.inv(K), x2)

                m1_T_m1 = np.matmul(m1.T, m1)
                m1_T_R_m2 = np.matmul(m1.T, (np.matmul(Ri.T, m2)))
                m2_T_m2 = np.matmul(m2.T, m2)

                t_T_m1 = np.matmul(ti.T, m1)
                t_T_R_m2 = np.matmul(ti.T, np.matmul(Ri.T, m2))

                LHS = [[m1_T_m1, -m1_T_R_m2], [m1_T_R_m2, m2_T_m2]]
                RHS = [t_T_m1, t_T_R_m2]

                # calculate the unknown distances 𝜆and 𝜇 by solving the linear equation
                x = np.linalg.solve(LHS, RHS)
                lambda_val = x[0]
                mu_val = x[1]

                # obtain the 3d coordinates of the scene points
                if lambda_val > 0 and mu_val > 0:
                    best_scene_points += 1
                    X_lambda = np.multiply(lambda_val, m1)
                    X_mu = np.add(ti, np.multiply(mu_val, np.matmul(Ri.T, m2)))
                    coord_3D_points.append((np.array(X_lambda), np.array(X_mu)))
            soln.append((Ri.T, ti))
            all_3D_points.append(coord_3D_points)
            scene_points_count.append(best_scene_points)

    # select the solution with most scene points that are in front of both frames
    # Discard all points, which are behind either of the frames for this solution as outliers
    best_soln = np.argmax(scene_points_count)
    best_t = np.array(soln[best_soln][1])
    best_R = np.array(soln[best_soln][0])
    best_3d_coords = all_3D_points[best_soln]

    all_frame1_points = []
    all_frame2_points = []

    for lambda_val, mu_val in best_3d_coords:
        all_frame1_points.append(lambda_val)
        all_frame2_points.append(mu_val)

    all_frame1_points = np.array(all_frame1_points)
    all_frame2_points = np.array(all_frame2_points)

    # Task 3 - Part D
    # Create a 3d plot to show the two camera centres and all 3d points
    plot_3d_coords = np.mean([all_frame1_points, all_frame2_points], axis=0)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(plot_3d_coords[:, 0], plot_3d_coords[:, 1], plot_3d_coords[:, 2], c='g', marker='o')
    ax.plot([0.], [0.], [0.], marker='X', c='b')
    ax.plot([best_t[0]], [best_t[1]], [best_t[2]], marker='X', c='b')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # Task 3E
    # Project the 3d points into the first and the last frame
    for i, pt in enumerate(plot_3d_coords):
        lambda_x = np.matmul(K, pt)
        x1 = lambda_x / lambda_x[2]

        cv2.circle(images[f1], (int(x1[1]), int(x1[0])), 2, (255, 0, 0), 2)
        # display the corresponding features to visualise the re-projection error
        cv2.circle(images[f1], (int(inliers[i][0][1]), int(inliers[i][0][0])), 2, (0, 255, 0), -1)

        cv2.line(images[f1], (int(x1[1]), int(x1[0])),
                 (int(inliers[i][0][1]), int(inliers[i][0][0])), (255, 255, 255), 1)

        mu_x = np.matmul(K, np.matmul(np.linalg.inv(best_R), (pt - best_t)))
        x2 = mu_x / mu_x[2]

        cv2.circle(images[f2], (int(x2[1]), int(x2[0])), 2, (255, 0, 0), 2)
        # display the corresponding features to visualise the re-projection error
        cv2.circle(images[f2], (int(inliers[i][1][1]), int(inliers[i][1][0])), 2, (0, 255, 0), -1)

        cv2.line(images[f2], (int(x2[1]), int(x2[0])),
                 (int(inliers[i][1][1]), int(inliers[i][1][0])), (255, 255, 255), 1)

    cv2.imshow("Frame 1 image", images[f1])
    cv2.imshow("Frame 2 image", images[f2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
