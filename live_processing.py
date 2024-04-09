import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Define the curve fitting function
def fitting_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


def fit_curve(xdata, ydata):
    initial_guesses = [1, 0.01, 0, 0]
    params, covariance = curve_fit(fitting_function, xdata, ydata, p0=initial_guesses)
    return params, covariance


def live_processing(vidObj, region):
    cv2.namedWindow('Processed Frame')
    H_values = []

    for _ in range(50):  # Process a fixed number of frames
        ret, frame = vidObj.read()
        if not ret:
            break

        crop_img = frame[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        adjusted_img = cv2.equalizeHist(gray_img)
        edges = cv2.Canny(adjusted_img, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fit curves to contours and compute height variation (simplified)
        if contours:
            for contour in contours:
                xdata = contour[:, :, 0].flatten()
                ydata = contour[:, :, 1].flatten()

                # Fit the curve to each contour
                try:
                    params, _ = fit_curve(xdata, ydata)

                    # Example of how to use the fitted curve
                    fitted_ydata = fitting_function(xdata, *params)
                    H = fitted_ydata.max() - fitted_ydata.min()
                    H_values.append(H)
                except Exception as e:
                    print(f"Curve fitting failed: {e}")
                    continue  # Skip to the next contour if fitting fails

        cv2.imshow('Processed Frame', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return max(H_values) if H_values else 0
