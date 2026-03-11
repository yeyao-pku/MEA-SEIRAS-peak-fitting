# logic/bl_correction.py

import numpy as np

def blcorrection(wavenumbers, intensities, mode="default", custom_range=None):
    """
    - mode: "default", "co", "water", "custom"
    - custom_range: tuple(start_idx, end_idx)
    """
    x = np.linspace(0, len(intensities) - 1, len(intensities))
    y_old = intensities.copy()
    k = np.linspace(0.000001, 0.0001, 1000)
    flag = 0
    i = 0
    now = 100

   
    if mode == "default":
        start, end = 0, 840 # 4000 cm^-1 to 2000 cm^-1 
    elif mode == "co":
        start, end = 981, 1401 # 2150 cm^-1 to 1800 cm^-1
    elif mode == "water":
        start, end = 0, 840 # 4000 cm^-1 to 2800 cm^-1
    elif mode == "custom":
        if custom_range and len(custom_range) == 2:
            start, end = custom_range
        else:
            raise ValueError("custom_range=(start, end) is needed")
    else:
        raise ValueError(f"unknown mode: {mode}")

    
    
    if abs((y_old[end] - y_old[start]) / y_old[start]) < 0.2:
        print(abs((y_old[end] - y_old[start]) / y_old[start]))
        y_new = y_old
        flag = 1
    else:
        while i < len(k):
            a = k[i]
            if (y_old[end] - y_old[start]) < 0:
                a = -a
            y_new = y_old - a * x
            if now < abs((y_new[end] - y_new[start]) / y_new[start]):
                if (y_old[end] - y_old[start]) < 0:
                    y_new = y_old + k[i - 1] * x
                else:
                    y_new = y_old - k[i - 1] * x
                flag = 1
                break
            i += 1
            now = abs((y_new[end] - y_new[start]) / y_new[start])
    
    if flag == 0:
        print("⚠️ failed to correct baseline, returning original intensities")
        y_new = y_old
    return y_new
