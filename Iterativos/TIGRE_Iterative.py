import numpy as np
import os
import sys
import gc
import time
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu
import matplotlib.pyplot as plt
import napari
import nibabel as nib
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HU_conversion import HU_conversion

from data_processing_3D_2 import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
    show_results,
)

def print_volume_info(volume, geo=None):

    print("\n" + "="*60)
    print("RECONSTRUCTED VOLUME INFORMATION")
    print("="*60)
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    print("="*60 + "\n") 
    print(f"\nGeometric information:")
    print(f"  - Voxel size: {geo.dVoxel} mm")
    print(f"  - Number of voxels: {geo.nVoxel}")
    print(f"  - Physical dimensions: {geo.sVoxel} mm")
    print("="*60 + "\n")


def export_volume_to_nii(volume, geo, source_folder, base_output="reconstructed_volumes"):
    dataset_name = os.path.basename(os.path.normpath(source_folder))
    now = datetime.now()
    timestamp = f"{now.day}_{now.strftime('%b').lower()}_{now.hour}h{now.minute}"
    
    folder_name = f"Iterative_{dataset_name}_{timestamp}"
    output_folder = os.path.join(base_output, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    filename = f"{dataset_name}.nii"
    filepath = os.path.join(output_folder, filename)
    
    volume_export = np.transpose(volume.astype(np.float32), (2, 1, 0))

    affine = np.eye(4)
    affine[0, 0] = geo.dVoxel[2]
    affine[1, 1] = geo.dVoxel[1]
    affine[2, 2] = geo.dVoxel[0]
    
    affine[0, 3] = -(volume_export.shape[0] * geo.dVoxel[2]) / 2.0
    affine[1, 3] = -(volume_export.shape[1] * geo.dVoxel[1]) / 2.0
    affine[2, 3] = -(volume_export.shape[2] * geo.dVoxel[0]) / 2.0
    
    nii_img = nib.Nifti1Image(volume_export, affine)
    nii_img.header.set_xyzt_units('mm', 'sec')
    nii_img.header['descrip'] = f'Iterative Reconstruction - {dataset_name}'
    
    print(f"    Saving NIfTI file...")
    nib.save(nii_img, filepath)
    print(f"    ✓ File saved: {filepath}")
    
    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Metadata - {timestamp}\n")
        f.write(f"Algorithm: Iterative\n")
        f.write(f"Geometry DSD: {geo.DSD}, DSO: {geo.DSO}\n")
        f.write(f"Voxel Size: {geo.dVoxel}\n")
        
        recon_info = globals().get('CONFIG')
        if recon_info:
            f.write("\nReconstruction Parameters:\n")
            for k, v in recon_info.items():
                f.write(f"  {k}: {v}\n")
                
    return filepath


def export_volume_HU(original_nii_path, volume, water_val, air_val):
    volume_HU = HU_conversion(volume, water_val, air_val)
    original_nii = nib.load(original_nii_path)
    
    volume_HU_export = np.transpose(volume_HU.astype(np.float32), (2, 1, 0))
    
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'
    
    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    print(f"    ✓ HU Volume saved: {os.path.basename(filepath_HU)}")
    return filepath_HU


def normalize_projections(projections_raw, I0_override=None):
    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 99))
    else:
        I0 = float(I0_override)

    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)
    ratio = np.clip(ratio, 1e-6, 1.2)
    
    projections_norm = -np.log(ratio)
    projections_norm[projections_norm < 0] = 0
    return projections_norm


def setup_geometry(img_shape, pixel_size, DSD, DSO, shift_pixels, total_angle, shift_sign):
    print(f"--> Setting up geometry. Shift: {shift_pixels:.2f} px (Sign: {shift_sign})")
    
    height, width, n_angles = img_shape
    geo = tigre.geometry(mode="cone")
    
    geo.nDetector = np.array([height, width])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    mag = DSD / DSO
    voxel_size = pixel_size / mag
    
    print(f"    Magnification: {mag:.4f}")
    print(f"    Voxel size: {voxel_size:.6f} mm")
    
    geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size])
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size),
        int(geo.sDetector[1] / voxel_size),
        int(geo.sDetector[1] / voxel_size)
    ])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.DSD = DSD
    geo.DSO = DSO
    
    shift_mm = shift_pixels * pixel_size
    geo.offDetector = np.array([0.0, shift_mm * shift_sign]) 
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    return geo, angles


def show_intermediate_slices(volume, iteration, algorithm_name):
    z, y, x = np.array(volume.shape) // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{algorithm_name} - Iteration {iteration}', fontweight='bold')
    
    axes[0].imshow(volume[z, :, :], cmap='gray'); axes[0].set_title('Axial')
    axes[1].imshow(volume[:, y, :], cmap='gray'); axes[1].set_title('Coronal')
    axes[2].imshow(volume[:, :, x], cmap='gray'); axes[2].set_title('Sagittal')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def reconstruct_with_progress(input_data, geo, angles, config):
    algorithm = config['algorithm'].upper()
    n_iter = config['n_iter']
    display_interval = config.get('display_interval', 20)
    

    sensitive_algs = ['FISTA', 'CGLS', 'ASD-POCS', 'IRN-TV-CGLS']
    must_run_continuous = any(sa in algorithm for sa in sensitive_algs)

    if must_run_continuous and display_interval < n_iter:
        print(f"\nAVISO: {algorithm} perde qualidade se for interrompido.")
        print("   -> Forçando execução contínua (sem update visual intermédio).")
        step = n_iter # Executa tudo de uma vez
    else:
        step = display_interval # Executa em blocos (ok para SIRT/SART)

    print(f"\n{'='*60}")
    print(f"STARTING RECONSTRUCTION: {algorithm}")
    print(f"Total Iterations: {n_iter}")
    print(f"{'='*60}\n")
    
    volumes_history = []
    volume = None 
    
    start_time = time.time()

    # Loop ajustado com o 'step' correto
    for batch_start in range(0, n_iter, step):
        batch_iter = min(step, n_iter - batch_start)
        current_total = batch_start + batch_iter
        
        print(f"--> Running iters {batch_start+1} to {current_total}...")
        
        # (O bloco de algoritmos manteve-se igual)
        if algorithm == 'SIRT':
            volume = algs.sirt(input_data, geo, angles, niter=batch_iter, init=volume)

        elif algorithm == 'OS-SART':
            blk = config.get('blocksize', 20)
            volume = algs.ossart(input_data, geo, angles, niter=batch_iter, blocksize=blk, init=volume)

        elif algorithm == 'CGLS':
            volume = algs.cgls(input_data, geo, angles, niter=batch_iter, init=volume)

        elif algorithm == 'ASD-POCS':
            volume = algs.asd_pocs(input_data, geo, angles, niter=batch_iter, init=volume,
                                   maxl2err=config.get('maxl2err', 0.15), 
                                   alpha=config.get('alpha', 0.002), 
                                   ng=config.get('ng', 25))
            
        elif algorithm == 'FISTA':
            volume = algs.fista(input_data, geo, angles, niter=batch_iter, init=volume,
                                hyper=config.get('hyper', 0.01))
            
        elif algorithm == 'B-ASD-POCS-BETA':
            volume = algs.b_asd_pocs_beta(input_data, geo, angles, niter=batch_iter, init=volume,
                                          maxl2err=config.get('maxl2err', 0.15),
                                          alpha=config.get('alpha', 0.002),
                                          tviter=config.get('tviter', 25),
                                          beta=config.get('beta', 1.0))
            
        elif algorithm == 'AWASD-POCS':
            volume = algs.awasd_pocs(input_data, geo, angles, niter=batch_iter, init=volume,
                                     maxl2err=config.get('maxl2err', 0.15),
                                     alpha=config.get('alpha', 0.002),
                                     tviter=config.get('tviter', 25))
            
        elif algorithm == 'IRN-TV-CGLS':
             volume = algs.irn_tv_cgls(input_data, geo, angles, niter=batch_iter, init=volume,
                                     lmbda=config.get('lmbda', 10))
             
        else:
            print(f"Warning: Algorithm '{algorithm}' not recognized. Defaulting to SIRT.")
            volume = algs.sirt(input_data, geo, angles, niter=batch_iter, init=volume)
        
        # Só mostra plot se não estiver no modo contínuo ou se for o final
        if not must_run_continuous or current_total == n_iter:
            show_intermediate_slices(volume, current_total, algorithm)
        
        # 2. FIX DE MEMÓRIA: Só guarda histórico se explicitamente pedido
        if config.get('show_history_in_napari', False):
            volumes_history.append(volume.copy())
            
        gc.collect()

    duration = time.time() - start_time
    print(f"\n--> Reconstruction Complete in {duration:.1f} seconds.")
    return volume, volumes_history


def main(tiff_folder, configurations):
    # 1. Carregar
    projections = load_images(tiff_folder)

    # 2. Downsample
    f = configurations['downsample']
    if f > 1:
        projections = projections[::f, ::f, :]
        pixel_size = configurations['pixel_size'] * f
    else:
        pixel_size = configurations['pixel_size']

    # 3. Ajuste do Shift
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / f
    print(f"--> Shift adjustment: {calibrated_shift_px} (original) -> {shift_val:.2f} px (binning {f})")

    # 4. Seleção de I0
    sino_raw = generate_collapsed_sinogram(projections)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background)
    
    projections_norm = normalize_projections(projections, I0_override=mean_I0)
    
    # 5. Visualizar Sinogramas
    sino_norm = generate_collapsed_sinogram(projections_norm)
    show_results(sino_raw, sino_norm, shift_val)
    
    # 3. FIX DE MEMÓRIA CRÍTICO: Limpar variáveis pesadas antes do passo mais pesado
    del projections, sino_raw, sino_norm
    gc.collect()

    # 6. Setup Geometria
    geo, angles = setup_geometry(
        projections_norm.shape, pixel_size,
        configurations['DSD'], configurations['DSO'],
        shift_val, configurations['total_angle'],
        shift_sign=configurations['shift_sign']
    )

    print("--> Transposing data for TIGRE...")
    input_data = np.transpose(projections_norm, (2, 0, 1)).astype(np.float32)
    
    # Limpa a variável original após o transpose (duplicação de memória evitada)
    del projections_norm
    gc.collect()
    
    # 7. Reconstrução Iterativa
    volume, volumes_history = reconstruct_with_progress(input_data, geo, angles, configurations)
    
    # 8. Visualização Final (Napari)
    print("--> Opening Napari...")
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]), name='Final_Volume')
    
    if configurations.get('show_history_in_napari', False):
        for idx, vol_hist in enumerate(volumes_history):
            # Se foi contínuo, o history pode só ter o ultimo, ajustamos a logica
            if len(volumes_history) == 1:
                 iter_num = configurations['n_iter']
            else:
                 iter_num = (idx + 1) * configurations['display_interval']
            
            viewer.add_image(vol_hist, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]), 
                           name=f'Iter_{iter_num}', visible=False)
    
    napari.run()
    
    # 9. Pós-Processamento e Exportação (Mantido igual)
    print_volume_info(volume, geo)
    
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    if export_choice == 'y':
        output_folder = export_volume_to_nii(volume, geo, tiff_folder)
        
        if input("\nAlso export in Hounsfield Units (HU)? (y/n): ").strip().lower() == 'y':
            w_val = float(input("  Enter gray scale value for WATER: "))
            a_val = float(input("  Enter gray scale value for AIR: "))
            export_volume_HU(output_folder, volume, w_val, a_val)

    return volume, volumes_history


if __name__ == "__main__":
    CONFIG = {
        'pixel_size': 0.05,   
        'DSD': 925,           
        'DSO': 893,           
        'total_angle': 2 * np.pi,  
        'downsample': 4,      
        'calibrated_shift_px': 5.12,
        'shift_sign': 1,      

        'algorithm': 'ASD-POCS',
        'n_iter': 75,           
        'display_interval': 75, 

        
        'blocksize': 20,          
        'maxl2err': 0.20,       
        'alpha': 0.002,           
        'tviter': 25,                
        'hyper': 0.01,            
        'lmbda': 10,              
        

        'export_nii': True,
        'show_history_in_napari': False,
    }
    
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    
    vol, history = main(folder, CONFIG)