# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(akilinc)s
"""

# -*- coding: utf-8 -*-
import numpy as np

class MeshCreator:
    def __init__(self, file_path, x_max, y_max, element_size, scale_factor=1.84):
        self.file_path = file_path
        self.nx_elems = int(round(x_max / element_size))
        self.ny_elems = int(round(y_max / element_size))
        self.nx_nodes = self.nx_elems + 1
        self.ny_nodes = self.ny_elems + 1
        self.element_size = element_size
        self.scale_factor = scale_factor

    def create_mesh_to_file(self):
        x = np.linspace(0, self.nx_elems * self.element_size, self.nx_nodes) * self.scale_factor
        y = np.linspace(0, self.ny_elems * self.element_size, self.ny_nodes) * self.scale_factor
        
        # order='F' ensures index [i, j] matches Abaqus X, Y
        node_ids = np.arange(1, self.nx_nodes * self.ny_nodes + 1).reshape((self.nx_nodes, self.ny_nodes), order='F')
        
        node_lines = []
        for i in range(self.nx_nodes):
            for j in range(self.ny_nodes):
                node_lines.append(f"{node_ids[i, j]}, {x[i]:.6f}, {y[j]:.6f}\n")

        element_ids = np.arange(1, self.nx_elems * self.ny_elems + 1).reshape((self.nx_elems, self.ny_elems), order='F')
        element_lines = []
        for i in range(self.nx_elems):
            for j in range(self.ny_elems):
                n1 = node_ids[i, j]
                n2 = node_ids[i+1, j]
                n3 = node_ids[i+1, j+1]
                n4 = node_ids[i, j+1]
                element_lines.append(f"{element_ids[i, j]}, {n1}, {n2}, {n3}, {n4}\n")

        def make_nset(name, ids):
            lines = [f"*Nset, nset={name}, instance=PART-1-1\n"]
            for k in range(0, len(ids), 16):
                lines.append(", ".join(map(str, ids[k:k+16])) + "\n")
            return lines

        node_sets = []
        node_sets.extend(make_nset("NODEBOTTOM", node_ids[:, 0]))
        node_sets.extend(make_nset("NODETOP", node_ids[:, -1]))
        node_sets.extend(make_nset("NODELEFT", node_ids[0, :]))
        node_sets.extend(make_nset("NODERIGHT", node_ids[-1, :]))

        with open(self.file_path, 'w') as f:
            f.write("*Heading\n** PARTS\n*Part, name=PART-1\n*NODE\n")
            f.writelines(node_lines)
            f.write("*Element, type=CPS4 \n")
            f.writelines(element_lines)
            
            for i in range(self.nx_elems):
                for j in range(self.ny_elems):
                    eid = element_ids[i, j]
                    f.write(f"*Elset, elset=Set-{eid}\n{eid}\n")
                    f.write(f"*Solid Section, elset=Set-{eid}, material=Material-{eid}\n")

            f.write("*End Part\n** ASSEMBLY\n*Assembly, name=Assembly\n")
            f.write("*Instance, name=PART-1-1, part=PART-1\n*End Instance\n")
            f.writelines(node_sets)
            f.write("*End Assembly\n")

class MaterialInserter:
    def __init__(self, file_path, disp_x_nodes, disp_y_nodes, stress_field_elems):
        self.file_path = file_path
        self.disp_x = disp_x_nodes  
        self.disp_y = disp_y_nodes  
        self.stress_field = stress_field_elems 

    def insert_data(self):
        nx_elems, ny_elems = self.stress_field.shape[1], self.stress_field.shape[2]
        nx_nodes, ny_nodes = self.disp_x.shape
        
        E_p = np.linspace(0.0, 0.2, 50)
        lines = []

        element_ids = np.arange(1, nx_elems * ny_elems + 1).reshape((nx_elems, ny_elems), order='F')
        lines.append("** MATERIALS\n")
        for i in range(nx_elems):
            for j in range(ny_elems):
                eid = element_ids[i, j]
                stress_vals = self.stress_field[:, i, j]
                
                lines.append(f"*Material, name=Material-{eid}\n")
                lines.append("*Elastic\n205000., 0.3\n*Plastic\n")
                stress_str = "\n".join([f"{stress_vals[k]:.6f}, {E_p[k]:.6f}" for k in range(50)])
                lines.append(f"{stress_str}\n")

        lines.extend([
            "** STEP: Step-1\n*Step, name=Step-1, nlgeom=NO, inc=1000\n",
            "*Static\n0.001, 1., 1e-06, 1.\n** BOUNDARY CONDITIONS\n"
        ])

        node_ids = np.arange(1, nx_nodes * ny_nodes + 1).reshape((nx_nodes, ny_nodes), order='F')

        def write_bc(name, n_id, u1, u2):
            return f"** Name: {name}_{n_id}\n*Boundary\nPART-1-1.{n_id}, 1, 1, {u1}\nPART-1-1.{n_id}, 2, 2, {u2}\n"

        for i in range(nx_nodes):
            lines.append(write_bc("Bottom", node_ids[i, 0], self.disp_x[i, 0], self.disp_y[i, 0]))
            lines.append(write_bc("Top", node_ids[i, -1], self.disp_x[i, -1], self.disp_y[i, -1]))

        for j in range(1, ny_nodes - 1):
            lines.append(write_bc("Left", node_ids[0, j], self.disp_x[0, j], self.disp_y[0, j]))
            lines.append(write_bc("Right", node_ids[-1, j], self.disp_x[-1, j], self.disp_y[-1, j]))

        lines.extend([
            "*Restart, write, frequency=0\n*Output, field\n*Node Output\nRF, U\n",
            "*Element Output, directions=YES\nE, PE, PEEQ, PEMAG, S\n",
            "*Output, history, variable=PRESELECT\n*End Step\n"
        ])

        with open(self.file_path, 'a') as f:
            f.writelines(lines)