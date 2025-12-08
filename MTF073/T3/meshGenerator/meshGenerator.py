# Clear all variables when running entire code:
from reset import universal_reset
universal_reset(protect={'universal_reset'}, verbose=True)

# Packages needed
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# ---------------------------------------------------------
# Flexible geometric stretching for one segment
# ---------------------------------------------------------
def flexible_geometric_stretch(n, a, b, r=1.2, mode='both'):
    L = b - a
    if mode == 'both':
        n_half = n // 2
        n_other = n - n_half
        if r == 1.0:
            sizes_half = np.full(n_half, L / n)
        else:
            sizes_half = np.zeros(n_half)
            sizes_half[0] = (L / 2) * (1 - r) / (1 - r**n_half)
            for i in range(1, n_half):
                sizes_half[i] = sizes_half[i-1] * r
        sizes_other = sizes_half[::-1]
        if n_other != n_half:
            sizes_other = np.concatenate((sizes_other, [sizes_other[-1]]))
        sizes = np.concatenate((sizes_half, sizes_other))
    elif mode == 'lower':
        if r == 1.0:
            sizes = np.full(n, L / n)
        else:
            sizes = np.zeros(n)
            sizes[0] = L * (1 - r) / (1 - r**n)
            for i in range(1, n):
                sizes[i] = sizes[i-1] * r
    elif mode == 'upper':
        if r == 1.0:
            sizes = np.full(n, L / n)
        else:
            sizes = np.zeros(n)
            sizes[-1] = L * (1 - r) / (1 - r**n)
            for i in range(n-2, -1, -1):
                sizes[i] = sizes[i+1] * r

    # Had an intermittent exception with the command below, so I exchanged it
    # with the following lines. Copilot said:
    # The line multiplies sizes in place. If sizes happens to have an integer
    # dtype (e.g., int64), and L / sizes.sum() is a float, NumPy will try to
    # write the float result back into an integer array. Modern NumPy correctly
    # refuses that unsafe cast and raises a UFuncTypeError like:
    # Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
    # Depending on your environment, this can sometimes surface with a
    # sparse/cryptic message (you mentioned “builtin”), but the root cause is
    # the same: in‑place arithmetic with mismatched dtypes.
    # Why it looks intermittent:
    # In some branches you create sizes with np.zeros(n) or np.full(..., float),
    # which are float arrays, so sizes *= float is fine.
    # In others, if the value passed to np.full or the way sizes is built leads
    # to an integer array (e.g., passing an integer literal or using integer
    # operations elsewhere), you’ll hit the casting error.
    # Small changes (commenting/uncommenting, re-running) can change code paths
    # or values and thus the inferred dtype, making it appear non-deterministic.
    #sizes *= L / sizes.sum()
    # First three lines just to make sure that we don't divide by zero:
    s = sizes.sum()
    if s == 0:
        raise ValueError("sizes.sum() is zero; cannot normalize.")
    # Main fix of the problem:
    sizes = sizes * (L / s)   # non in-place to allow safe upcast
    # Other options:
    # Force float dtype before scaling:
    # sizes = sizes.astype(float)
    # sizes *= L / sizes.sum()
    # Create arrays as float from the start (recommended):
    # sizes = np.zeros(n, dtype=float)

    faces = np.concatenate(([a], a + np.cumsum(sizes)))
    return faces

# ---------------------------------------------------------
# Multi-segment mesh generator
# ---------------------------------------------------------
def multi_segment_mesh(segments):
    faces = []
    for seg in segments:
        a, b = seg['start'], seg['end']
        n = seg['cells']
        r = seg['r']
        mode = seg['mode']
        seg_faces = flexible_geometric_stretch(n, a, b, r, mode)
        if faces:
            seg_faces = seg_faces[1:]  # avoid duplicate point
        faces.extend(seg_faces)
    return np.array(faces)


# def show_custom_message(msg):
#     root = tk.Tk()
#     root.withdraw()  # Hide main window
#     win = tk.Toplevel()
#     win.title("Message")
#     ttk.Label(win, text=msg, padding=10).pack()
#     ttk.Button(win, text="OK", command=win.destroy).pack(pady=5)
#     win.mainloop()

# def show_custom_error(msg):
#     root = tk.Tk()
#     root.withdraw()  # Hide main window
#     win = tk.Toplevel()
#     win.title("Error")
#     ttk.Label(win, text=msg, padding=10).pack()
#     ttk.Button(win, text="OK", command=win.destroy).pack(pady=5)
#     win.mainloop()

# ---------------------------------------------------------
# GUI Class
# ---------------------------------------------------------
class MeshGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Generator")

        # Domain inputs
        tk.Label(root, text="Domain X (start,end):").grid(row=0, column=0)
        self.x_start = tk.Entry(root)
        self.x_start.insert(0, "0.0")
        self.x_start.grid(row=0, column=1)
        self.x_end = tk.Entry(root)
        self.x_end.insert(0, "1.0")
        self.x_end.grid(row=0, column=2)

        tk.Label(root, text="Domain Y (start,end):").grid(row=1, column=0)
        self.y_start = tk.Entry(root)
        self.y_start.insert(0, "0.0")
        self.y_start.grid(row=1, column=1)
        self.y_end = tk.Entry(root)
        self.y_end.insert(0, "1.0")
        self.y_end.grid(row=1, column=2)

        # Segments count
        tk.Label(root, text="Number of X segments:").grid(row=2, column=0)
        self.x_segments = tk.Entry(root)
        self.x_segments.insert(0, "1")
        self.x_segments.grid(row=2, column=1)

        tk.Label(root, text="Number of Y segments:").grid(row=3, column=0)
        self.y_segments = tk.Entry(root)
        self.y_segments.insert(0, "1")
        self.y_segments.grid(row=3, column=1)

        # Buttons for configuration and loading
        tk.Button(root, text="Configure Segments", command=self.configure_segments).grid(row=4, column=0, columnspan=2, sticky="ew")
        tk.Button(root, text="Load Settings", command=self.load_settings_main).grid(row=4, column=2, sticky="ew")

        self.segment_window = None  # Track if a config window is open

    # -----------------------------------------------------
    # Configure segments window
    # -----------------------------------------------------
    def configure_segments(self):
        if self.segment_window and tk.Toplevel.winfo_exists(self.segment_window):
            messagebox.showinfo("Info", "A configuration window is already open.")
            return

        self.create_segment_window()

    def create_segment_window(self):
        self.segment_window = tk.Toplevel(self.root)
        self.segment_window.title("Segment Configuration")

        try:
            self.num_x = int(self.x_segments.get())
            self.num_y = int(self.y_segments.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid segment count")
            return

        # Column headers
        headers = ["Start", "End", "Cells", "Ratio", "Mode"]
        for col, header in enumerate(headers, start=1):
            tk.Label(self.segment_window, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col)

        self.x_entries = []
        self.y_entries = []

        # X Segments
        tk.Label(self.segment_window, text="X Segments").grid(row=0, column=0)
        for i in range(self.num_x):
            tk.Label(self.segment_window, text=f"X{i+1}").grid(row=i+2, column=0)
            start = tk.Entry(self.segment_window)
            if i == 0:
                start.insert(0, self.x_start.get())
            else:
                start.insert(0, "")
                start.config(state='disabled')
            end = tk.Entry(self.segment_window)
            cells = tk.Entry(self.segment_window)
            r = tk.Entry(self.segment_window)
            mode = ttk.Combobox(self.segment_window, values=['both','lower','upper'])
            mode.set('both')
            end_val = float(self.x_start.get()) + (i+1)*(float(self.x_end.get())-float(self.x_start.get()))/self.num_x
            end.insert(0, str(end_val))
            cells.insert(0, "20")
            r.insert(0, "1.2")
            start.grid(row=i+2, column=1)
            end.grid(row=i+2, column=2)
            cells.grid(row=i+2, column=3)
            r.grid(row=i+2, column=4)
            mode.grid(row=i+2, column=5)
            self.x_entries.append((start,end,cells,r,mode))

        # Y Segments
        offset = self.num_x + 3
        tk.Label(self.segment_window, text="Y Segments").grid(row=offset, column=0)
        for jdx in range(self.num_y):
            tk.Label(self.segment_window, text=f"Y{jdx+1}").grid(row=offset+jdx+1, column=0)
            start = tk.Entry(self.segment_window)
            if jdx == 0:
                start.insert(0, self.y_start.get())
            else:
                start.insert(0, "")
                start.config(state='disabled')
            end = tk.Entry(self.segment_window)
            cells = tk.Entry(self.segment_window)
            r = tk.Entry(self.segment_window)
            mode = ttk.Combobox(self.segment_window, values=['both','lower','upper'])
            mode.set('both')
            end_val = float(self.y_start.get()) + (jdx+1)*(float(self.y_end.get())-float(self.y_start.get()))/self.num_y
            end.insert(0, str(end_val))
            cells.insert(0, "20")
            r.insert(0, "1.2")
            start.grid(row=offset+jdx+1, column=1)
            end.grid(row=offset+jdx+1, column=2)
            cells.grid(row=offset+jdx+1, column=3)
            r.grid(row=offset+jdx+1, column=4)
            mode.grid(row=offset+jdx+1, column=5)
            self.y_entries.append((start,end,cells,r,mode))

        # Buttons in one centered row
        row_buttons = offset + self.num_y + 1
        for col in range(4):
            self.segment_window.grid_columnconfigure(col, weight=1)

        tk.Button(self.segment_window, text="Generate Mesh", command=self.generate_mesh).grid(row=row_buttons, column=1, sticky="ew")
        tk.Button(self.segment_window, text="Close Figures", command=lambda: plt.close('all')).grid(row=row_buttons, column=2, sticky="ew")
        tk.Button(self.segment_window, text="Save Settings", command=self.save_settings).grid(row=row_buttons, column=3, sticky="ew")
        tk.Button(self.segment_window, text="Save Mesh", command=self.export_mesh_npz).grid(row=row_buttons, column=4, sticky="ew")

    # -----------------------------------------------------
    # Load Settings from main window
    # -----------------------------------------------------
    def load_settings_main(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv")],
                                               title="Load Mesh Settings")
        if not file_path:
            return

        # If a config window is open, ask user if it should be closed
        if self.segment_window and tk.Toplevel.winfo_exists(self.segment_window):
            if not messagebox.askyesno("Close Window?", "A configuration window is open. Close it and load new settings?"):
                return
            self.segment_window.destroy()

        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            x_lines = [line for line in lines if line.startswith("X") and not line.startswith("X Segments")]
            y_lines = [line for line in lines if line.startswith("Y") and not line.startswith("Y Segments")]

            self.x_segments.delete(0, tk.END)
            self.x_segments.insert(0, str(len(x_lines)))
            self.y_segments.delete(0, tk.END)
            self.y_segments.insert(0, str(len(y_lines)))

            self.create_segment_window()

            prev_end_x = None
            for idx, line in enumerate(x_lines):
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip() for p in line.split(',')}
                start_val = parts.get('start', '')
                end_val = parts.get('end', '')
                cells_val = parts.get('cells', '10')
                ratio_val = parts.get('ratio', '1.0')
                mode_val = parts.get('mode', 'both')
                if start_val == '':
                    start_val = prev_end_x if prev_end_x else self.x_start.get()
                prev_end_x = end_val
                self.x_entries[idx][0].delete(0, tk.END)
                self.x_entries[idx][0].insert(0, start_val)
                self.x_entries[idx][1].delete(0, tk.END)
                self.x_entries[idx][1].insert(0, end_val)
                self.x_entries[idx][2].delete(0, tk.END)
                self.x_entries[idx][2].insert(0, cells_val)
                self.x_entries[idx][3].delete(0, tk.END)
                self.x_entries[idx][3].insert(0, ratio_val)
                self.x_entries[idx][4].set(mode_val)

            prev_end_y = None
            for jdx, line in enumerate(y_lines):
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip() for p in line.split(',')}
                start_val = parts.get('start', '')
                end_val = parts.get('end', '')
                cells_val = parts.get('cells', '10')
                ratio_val = parts.get('ratio', '1.0')
                mode_val = parts.get('mode', 'both')
                if start_val == '':
                    start_val = prev_end_y if prev_end_y else self.y_start.get()
                prev_end_y = end_val
                self.y_entries[jdx][0].delete(0, tk.END)
                self.y_entries[jdx][0].insert(0, start_val)
                self.y_entries[jdx][1].delete(0, tk.END)
                self.y_entries[jdx][1].insert(0, end_val)
                self.y_entries[jdx][2].delete(0, tk.END)
                self.y_entries[jdx][2].insert(0, cells_val)
                self.y_entries[jdx][3].delete(0, tk.END)
                self.y_entries[jdx][3].insert(0, ratio_val)
                self.y_entries[jdx][4].set(mode_val)

            # With system sound:
            messagebox.showinfo("Success", f"Settings loaded from {file_path}")
            # Without system sound:
            # show_custom_message(f"Success: Settings loaded from {file_path}")
        except Exception as e:
            # With system sound:
            messagebox.showerror("Error", f"Failed to load settings: {e}")
            # Without system sound:
            # show_custom_error(f"Error: Failed to load settings: {e}")

    # -----------------------------------------------------
    # Collect segments
    # -----------------------------------------------------
    def collect_segments(self):
        segments_x, segments_y = [], []
        prev_end_x = None
        for idx,(start,end,cells,r,mode) in enumerate(self.x_entries):
            seg_start = float(start.get()) if idx == 0 else prev_end_x
            seg_end = float(end.get())
            if seg_end <= seg_start:
                raise ValueError(f"X segment {idx+1}: End point must be greater than start point.")
            prev_end_x = seg_end
            segments_x.append({'start':seg_start,'end':seg_end,'cells':int(cells.get()),'r':float(r.get()),'mode':mode.get()})

        prev_end_y = None
        for jdx,(start,end,cells,r,mode) in enumerate(self.y_entries):
            seg_start = float(start.get()) if jdx == 0 else prev_end_y
            seg_end = float(end.get())
            if seg_end <= seg_start:
                raise ValueError(f"Y segment {jdx+1}: End point must be greater than start point.")
            prev_end_y = seg_end
            segments_y.append({'start':seg_start,'end':seg_end,'cells':int(cells.get()),'r':float(r.get()),'mode':mode.get()})

        return segments_x, segments_y

    # -----------------------------------------------------
    # Generate mesh and plot
    # -----------------------------------------------------
    def generate_mesh(self):
        try:
            segments_x, segments_y = self.collect_segments()
            x_faces = multi_segment_mesh(segments_x)
            y_faces = multi_segment_mesh(segments_y)
            pointX, pointY = np.meshgrid(x_faces, y_faces, indexing='ij')

            self.last_pointX = pointX
            self.last_pointY = pointY

            plt.figure(figsize=(7,7))
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.title('Generated Computational Mesh')
            plt.axis('equal')
            plt.vlines(pointX[:,0], pointY[0,0], pointY[0,-1], colors='k')  # solid lines
            plt.hlines(pointY[0,:], pointX[0,0], pointX[-1,0], colors='k')  # solid lines
            plt.show()
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate mesh: {e}")

    # -----------------------------------------------------
    # Save Settings
    # -----------------------------------------------------
    def save_settings(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv")],
                                                 title="Save Mesh Settings")
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                f.write("X Segments:\n")
                for idx,(start,end,cells,r,mode) in enumerate(self.x_entries):
                    f.write(f"X{idx+1}: start={start.get()}, end={end.get()}, cells={cells.get()}, ratio={r.get()}, mode={mode.get()}\n")
                f.write("\nY Segments:\n")
                for jdx,(start,end,cells,r,mode) in enumerate(self.y_entries):
                    f.write(f"Y{jdx+1}: start={start.get()}, end={end.get()}, cells={cells.get()}, ratio={r.get()}, mode={mode.get()}\n")
            messagebox.showinfo("Success", f"Settings saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    # -----------------------------------------------------
    # Export Mesh as .npz
    # -----------------------------------------------------
    def export_mesh_npz(self):
        if not hasattr(self, 'last_pointX') or not hasattr(self, 'last_pointY'):
            messagebox.showerror("Error", "No mesh generated yet. Please generate the mesh first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".npz",
                                                 filetypes=[("NumPy Files", "*.npz")],
                                                 title="Save Mesh")
        if not file_path:
            return
        try:
            np.savez(file_path, pointX = self.last_pointX, pointY = self.last_pointY)
            messagebox.showinfo("Success", f"Mesh saved as NumPy file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mesh: {e}")

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
root = tk.Tk()
app = MeshGUI(root)
root.mainloop()
