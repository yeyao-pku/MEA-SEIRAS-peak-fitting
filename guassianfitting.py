import os
import numpy as np
#np.random.seed(42) 
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from numpy import trapz 
import glob
import importlib.util
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def three_gaussian_fit(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3):
    return (gaussian(x, a1, mu1, sigma1) + 
            gaussian(x, a2, mu2, sigma2) + 
            gaussian(x, a3, mu3, sigma3))


def read_dpt(filepath):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]

def load_blcorrection():
    # please adjust the path to where your blcorrection.py is located
    target_dir = "/Users/valentina/Desktop/peak-fitting"
    if os.path.exists(target_dir):
        os.chdir(target_dir)
    for file in os.listdir():
        if file.lower() == 'blcorrection.py':
            path = os.path.abspath(file)
            spec = importlib.util.spec_from_file_location("blcorrection", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.blcorrection
    return None 

# ================= dynamic fitting =================

def fit_water_peaks_dynamic(wavenumbers, intensities, initial_guess, current_bounds):

    y_corrected = intensities.copy()
    offset1 = y_corrected[0]
    offset2 = y_corrected[840] if len(y_corrected) > 840 else y_corrected[-1]
    k = (offset2 - offset1) / 840
    x_idx = np.arange(len(y_corrected))
    yy = offset1 + k * x_idx
    y_corrected = y_corrected - yy

    mask = (wavenumbers >= 2800) & (wavenumbers <= 4000)
    x_fit = wavenumbers[mask]
    y_fit = y_corrected[mask]

    
    def objective(params):
        return np.sum((three_gaussian_fit(x_fit, *params) - y_fit)**2)

    bound_pairs = list(zip(current_bounds[0], current_bounds[1]))

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bound_pairs,
        "args": ()
    }

    res = basinhopping(objective, initial_guess, niter=50, 
                       minimizer_kwargs=minimizer_kwargs, stepsize=15)
    popt = res.x


    residuals = y_fit - three_gaussian_fit(x_fit, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r_squared = 1 - ss_res / ss_tot

    peak_areas = []
    peak_params = []
    for i in range(3):
        a, mu, sigma = popt[3*i], popt[3*i+1], popt[3*i+2]
        
        x_fine = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        y_fine = gaussian(x_fine, a, mu, sigma)
        area = trapz(y_fine, x_fine)
        peak_areas.append(area)
        peak_params.append((a, mu, sigma))

    return y_corrected, popt, peak_params, peak_areas, r_squared, x_fit, y_fit

# ================= plotting =================

def add_scale_bar(ax, scale_length=0.005):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y_pos = ylim[1] - 0.1 * (ylim[1] - ylim[0])
    ax.plot([x_pos, x_pos], [y_pos - scale_length, y_pos], color='black', lw=1.5)
    ax.text(x_pos + 50, y_pos - scale_length/2, f'{scale_length} a.u.', va='center')

def plot_results_all_together(all_data, output_filename="dynamic_fit_results.png"):
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    fig, ax = plt.subplots(figsize=(5, 10))
    
    N = len(all_data)
    cmap = LinearSegmentedColormap.from_list('fit_cmap', ['#00008B', '#8B0000'], N=N)
    
    vertical_offset = 0
    for i, (filename, wavenumbers, _, y_corr, popt, peak_params, _, r_sq) in enumerate(all_data):
        mask = (wavenumbers >= 2800) & (wavenumbers <= 4000)
        x_disp = wavenumbers[mask]
        y_disp = y_corr[mask] + vertical_offset
        
        ax.plot(x_disp, y_disp, color=cmap(i/N), alpha=0.4, lw=1)
        
        y_fit = three_gaussian_fit(x_disp, *popt) + vertical_offset
        ax.plot(x_disp, y_fit, '--', color=cmap(i/N), lw=1.5)
        
        colors = ['#B61E2E', '#AFCFE2', '#132F5E']
        for j, (a, mu, sigma) in enumerate(peak_params):
            y_p = gaussian(x_disp, a, mu, sigma) + vertical_offset
            ax.fill_between(x_disp, vertical_offset, y_p, color=colors[j], alpha=0.3, lw=0)
            
        vertical_offset += 0.01 

    ax.set_xlim(4000, 2800)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Absorbance (a.u.)')
    ax.set_yticks([])
    add_scale_bar(ax)
    plt.tight_layout()
    # adjust the path to where you want to save the figure
    os.chdir("/Users/valentina/Desktop/peak-fitting")
    plt.savefig(output_filename, dpi=300)
    plt.show()



def main():
    
    try:
        blcorrection_func = load_blcorrection()
        print("Baseline correction function loaded successfully.")
    except Exception as e:
        print(f"Error loading baseline correction function: {e}")
        return
    
    # adjust the path to where your .dpt files are located   
    base_path = "/Users/valentina/Desktop/mea-seiras"
    os.chdir(base_path)

    folders = sorted([f for f in os.listdir() if os.path.isdir(f)])
    print("\nFolders in directory:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    folder_choice = input("\nEnter folder index (comma-separated for multiple, 'a' for all): ").strip()
    selected_folders = folders if folder_choice.lower() == 'a' else [folders[int(i) - 1] for i in folder_choice.split(',')]

    dpt_files = []
    for folder in selected_folders:
        dpt_files.extend(glob.glob(os.path.join(folder, '*.dpt')))
    dpt_files.sort()

    if not dpt_files:
        print("No .dpt files found.")
        return

    print("\nFound .dpt files:")
    for i, file in enumerate(dpt_files, 1):
        print(f"{i}. {file}")
    
    file_choice = input("\nEnter file index (comma-separated for multiple, 'a' for all): ").strip()
    selected_files = dpt_files if file_choice.lower() == 'a' else [dpt_files[int(i) - 1] for i in file_choice.split(',')]

   
    # 1. 初始化参数（针对第一张谱图）
    # [a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3
    initial_guess = [ 0.005, 3100,120,  -0.003, 3300, 120, 0.01, 3500, 100]
    # 初始宽边界
    low_b = [-0.5, 3000, 30,  -0.5, 3200, 40,  -0.5, 3600, 20]
    high_b = [0.5, 3200, 120,  0.5, 3600, 120,  0.5, 3800, 100]
    current_bounds = (low_b, high_b)

    all_plot_data = []
    detailed_results = []

    
    print(f"\nStarting dynamic fitting for {len(selected_files)} files...")

    for i, filepath in enumerate(selected_files):
        filename = os.path.basename(filepath)
        try:
            wavenumbers, intensities = read_dpt(filepath)
            
            y_corr, popt, p_params, p_areas, r2, x_fit, y_fit = \
                fit_water_peaks_dynamic(wavenumbers, intensities, initial_guess, current_bounds)
            
            print(f"[{i+1}/{len(selected_files)}] {filename} | R²: {r2:.4f}")

            print("-" * 30)
            print(f"Fitting details for {filename}:")
            peak_names = ['Ice-like', 'Hydration', 'Isolated']

            for j in range(3):
                a, mu, sigma = popt[3*j : 3*j+3]
                fwhm = 2.355 * sigma
                print(f"  Peak {j+1} ({peak_names[j]}):")
                print(f"    Amplitude (a):     {a:.6f}")
                print(f"    Center (mu):    {mu:.2f} cm-1")
                print(f"    FWHM: {fwhm:.2f} cm-1")
            print("-" * 30)
            
        
            initial_guess = popt 
            
        
            new_low = []
            new_high = []
            for idx, val in enumerate(popt):
                if idx % 3 == 0:   
                    n_l = val - abs(val) * 0.8
                    n_h = val + abs(val) * 0.8
                    if val < 0:
                        n_h = min(n_h, 0)
                
                    new_low.append(n_l)
                    new_high.append(n_h)

                elif idx % 3 == 1: 
                    new_low.append(val - 10)
                    new_high.append(val + 10)
                else:
                    new_low.append(low_b[idx])
                    new_high.append(high_b[idx])
            
            current_bounds = (new_low, new_high)
            
            all_plot_data.append((filename, wavenumbers, intensities, y_corr, popt, p_params, p_areas, r2))
            
            total_area = sum(p_areas)
            
            detailed_results.append({
                'Filename': filename,
                'R²': r2,
                
                # Peak 1
                'ice_like_water_area': p_areas[0],
                'ice_like_water_center': p_params[0][1],
                'ice_like_water_amplitude': p_params[0][0],
                'ice_like_water_sigma': p_params[0][2],
                'ice_like_water_fwhm': 2.355 * p_params[0][2],
                
                # Peak 2
                'hydration_water_area': p_areas[1],
                'hydration_water_center': p_params[1][1],
                'hydration_water_amplitude': p_params[1][0],
                'hydration_water_sigma': p_params[1][2],
                'hydration_water_fwhm': 2.355 * p_params[1][2],
                
                # Peak 3
                'isolated_water_area': p_areas[2],
                'isolated_water_center': p_params[2][1],
                'isolated_water_amplitude': p_params[2][0],
                'isolated_water_sigma': p_params[2][2],
                'isolated_water_fwhm': 2.355 * p_params[2][2],
                
                'baseline_k': 0, 
                'baseline_b': 0,
                
                'total_area': total_area,

                'abs_total_area': sum(abs(a) for a in p_areas), 
                'ice_like_water_ratio': p_areas[0] / total_area if total_area != 0 else 0,
                'hydration_water_ratio': p_areas[1] / total_area if total_area != 0 else 0,
                'isolated_water_ratio': p_areas[2] / total_area if total_area != 0 else 0
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")


    os.chdir("/Users/valentina/Desktop/peak-fitting")
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv("summary_dynamic_fit_2026.csv", index=False)
        print(f"\nResults successfully saved to summary_dynamic_fit_2026.csv")

    if all_plot_data:
        plot_results_all_together(all_plot_data)

if __name__ == '__main__':
    main()