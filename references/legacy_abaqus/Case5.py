# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(akilinc)s
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

path = r"C:/Users/adil.kilinc/Desktop/Thesis/"
os.chdir(os.path.join(path, r"Case_1_Pixel/3_code/Input_file_generators"))
import generate_mesh_no_interp_2  as mesh_gen #_uniaxial

disp_y = np.load(os.path.join(path, r"3_data\22_U_V_again_\U_40.npy")) * 0.00184
disp_x = np.load(os.path.join(path, r"3_data\22_U_V_again_\V_40.npy")) * 0.00184

x_size, y_size = 0.1, 0.1
element_size = 0.001 
scale_factor = 1.84 

nx_elems = int(x_size / element_size)
ny_elems = int(y_size / element_size)

def crop_center(array, n_rows, n_cols):
    start_row = (array.shape[0] - n_rows) // 2
    start_col = (array.shape[1] - n_cols) // 2
    return array[start_row:start_row + n_rows, start_col:start_col + n_cols]

# From raw (3600, 3100): crop 101 rows (perp), 1001 cols (tensile)
center_disp_y = crop_center(disp_y, ny_elems + 1, nx_elems + 1)
center_disp_x = crop_center(disp_x, ny_elems + 1, nx_elems + 1)


num_of_elements = int(x_size*y_size/(element_size**2))


if num_of_elements >= 1_000_000:
    num_elements_str = f"{int(num_of_elements / 1_000_000)}m"
elif num_of_elements >= 1_000:
    num_elements_str = f"{int(num_of_elements / 1_000)}k"
else:
    num_elements_str = str(int(num_of_elements))


@jit
def calc_stress(yielding, K, n):
    E_p = jnp.linspace(0.0, 0.2, 50)
    return jax.vmap(lambda e_p: yielding + K * jnp.power(e_p, n))(E_p)

# Material arrays loaded purely (no transpose)
yield_pixel = np.load(os.path.join(path, f"3_data/25/el_thresh50.npy"))
K_pixel = np.load(os.path.join(path, f"3_data/25/Hardening_coeff_el_Thresh50.npy")) * 396

yield_cropped = crop_center(yield_pixel, nx_elems, ny_elems)
K_cropped = crop_center(K_pixel, nx_elems, ny_elems)

stresses = np.array(calc_stress(yield_cropped, K_cropped, 0.245))

out_dir = os.path.join(path, "Case_1_Pixel/2_inp/plasticity")
os.makedirs(out_dir, exist_ok=True)
filename = f"mesh_central_test_thr50_{num_elements_str}_semi_uniaxial.inp"
file_path = os.path.join(out_dir, filename)

mc = mesh_gen.MeshCreator(file_path, x_size, y_size, element_size, scale_factor)
mc.create_mesh_to_file()

mi = mesh_gen.MaterialInserter(file_path, center_disp_x, center_disp_y, stresses)
mi.insert_data()
print("Input file generated successfully.")