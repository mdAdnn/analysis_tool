from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Global variables to store the nucleus positions for control and experimental groups
control_nucleus_positions = []
experimental_nucleus_positions = []

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            control_image = request.files['control_image']
            experimental_image = request.files['experimental_image']

            control_image.save(os.path.join('static/uploaded_images', control_image.filename))
            experimental_image.save(os.path.join('static/uploaded_images', experimental_image.filename))

            return render_template('annotate.html', control_image=control_image.filename, experimental_image=experimental_image.filename)
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'POST':
        control_positions = request.form.getlist('control_positions')
        experimental_positions = request.form.getlist('experimental_positions')

        try:
            # Check if positions are selected
            if not control_positions or not experimental_positions:
                raise ValueError("No positions selected. Please select points on the images.")

            control_nucleus_positions = [(int(pos.split(',')[0]), int(pos.split(',')[1])) for pos in control_positions if pos]
            experimental_nucleus_positions = [(int(pos.split(',')[0]), int(pos.split(',')[1])) for pos in experimental_positions if pos]

            if not control_nucleus_positions or not experimental_nucleus_positions:
                raise ValueError("No valid positions found. Please select points on the images.")

            run_number = request.form['run_number']
            scale_mm_per_pixel = float(request.form['scale_mm_per_pixel'])

            df_control_positions = pd.DataFrame(control_nucleus_positions, columns=['X', 'Y'])
            df_experimental_positions = pd.DataFrame(experimental_nucleus_positions, columns=['X', 'Y'])

            excel_control_file_name = f'static/results/control_nucleus_positions_run{run_number}.xlsx'
            excel_experimental_file_name = f'static/results/experimental_nucleus_positions_run{run_number}.xlsx'

            df_control_positions.to_excel(excel_control_file_name, index=False)
            df_experimental_positions.to_excel(excel_experimental_file_name, index=False)

            merged_positions_df = pd.concat([df_control_positions, df_experimental_positions], ignore_index=True)
            merged_positions_df['Group'] = ['Control'] * len(df_control_positions) + ['Experimental'] * len(df_experimental_positions)

            excel_merged_file_name = f'static/results/merged_nucleus_positions_run{run_number}.xlsx'
            merged_positions_df.to_excel(excel_merged_file_name, index=False)

            control_distances = squareform(pdist(control_nucleus_positions))
            experimental_distances = squareform(pdist(experimental_nucleus_positions))

            control_df_distances = pd.DataFrame(control_distances, columns=[f'Nucleus_{i}' for i in range(len(control_distances))])
            experimental_df_distances = pd.DataFrame(experimental_distances, columns=[f'Nucleus_{i}' for i in range(len(experimental_distances))])

            control_df_distances.to_excel('static/results/control_distances.xlsx', index=False)
            experimental_df_distances.to_excel('static/results/experimental_distances.xlsx', index=False)

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))

            control_image_rgb = cv2.imread(os.path.join('static/uploaded_images', request.form['control_image']))
            control_image_rgb = cv2.cvtColor(control_image_rgb, cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(control_image_rgb)
            axs[0, 0].set_title('Control Image')
            axs[0, 0].axis('off')
            for position in control_nucleus_positions:
                axs[0, 0].plot(position[0], position[1], marker='o', markersize=6, color='green')

            experimental_image_rgb = cv2.imread(os.path.join('static/uploaded_images', request.form['experimental_image']))
            experimental_image_rgb = cv2.cvtColor(experimental_image_rgb, cv2.COLOR_BGR2RGB)
            axs[0, 1].imshow(experimental_image_rgb)
            axs[0, 1].set_title('Experimental Image')
            axs[0, 1].axis('off')
            for position in experimental_nucleus_positions:
                axs[0, 1].plot(position[0], position[1], marker='o', markersize=6, color='yellow')

            axs[0, 2].scatter(range(len(control_distances.flatten())), control_distances.flatten(), color='green', label='Control', alpha=0.5)
            axs[0, 2].scatter(range(len(experimental_distances.flatten())), experimental_distances.flatten(), color='blue', label='Experimental', alpha=0.5)
            axs[0, 2].set_xlabel('Pair Index')
            axs[0, 2].set_ylabel('Distance (mm)')
            axs[0, 2].set_title('Scatter Plot of Nucleus Distances')
            axs[0, 2].legend()

            axs[1, 0].hist(control_distances.flatten(), bins=20, color='green', alpha=0.5, label='Control')
            axs[1, 0].set_xlabel('Distance (mm)')
            axs[1, 0].set_ylabel('Frequency')
            axs[1, 0].set_title('Histogram of Nucleus Distances (Control)')
            axs[1, 0].legend()

            axs[1, 1].hist(experimental_distances.flatten(), bins=20, color='blue', alpha=0.5, label='Experimental')
            axs[1, 1].set_xlabel('Distance (mm)')
            axs[1, 1].set_ylabel('Frequency')
            axs[1, 1].set_title('Histogram of Nucleus Distances (Experimental)')
            axs[1, 1].legend()

            axs[1, 2].hist(control_distances.flatten(), bins=20, color='green', alpha=0.5, label='Control')
            axs[1, 2].hist(experimental_distances.flatten(), bins=20, color='blue', alpha=0.5, label='Experimental')
            axs[1, 2].set_xlabel('Distance (mm)')
            axs[1, 2].set_ylabel('Frequency')
            axs[1, 2].set_title('Combined Histogram of Nucleus Distances')
            axs[1, 2].legend()

            plt.tight_layout()
            plot_path = f'static/results/nucleus_distances_run{run_number}.png'
            plt.savefig(plot_path)
            plt.close()

            return render_template('results.html', plot_path=plot_path, excel_merged_file_name=excel_merged_file_name)
        
        except ValueError as e:
            flash(str(e))
            return redirect(url_for('annotate', control_image=request.form['control_image'], experimental_image=request.form['experimental_image']))
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(url_for('annotate', control_image=request.form['control_image'], experimental_image=request.form['experimental_image']))
    
    control_image = request.args.get('control_image')
    experimental_image = request.args.get('experimental_image')
    return render_template('annotate.html', control_image=control_image, experimental_image=experimental_image)

if __name__ == "__main__":
    app.run(debug=True)
