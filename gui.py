import gradio as gr
import simpy
import random
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from simulation import QueueSystem
import simulation as sm
import certifi
import seaborn as sns
import matplotlib.animation as animation
from io import BytesIO
import os
import base64
import openpyxl
from PIL import Image

def create_queue_animation(queue_lengths, simulation_time):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(queue_lengths))
    ax.set_ylim(0, max(queue_lengths) + 5)
    ax.set_title("Queue Length Per Customer")
    ax.set_xlabel("Customer")
    ax.set_ylabel("Queue Length")
    line, = ax.plot([], [], lw=2, color="blue", label="Queue Length")
    ax.legend()

    frames = []

    for frame in range(len(queue_lengths)):
        x = list(range(frame))
        y = queue_lengths[:frame]
        line.set_data(x, y)

        buf = BytesIO()
        plt.savefig(buf, format='png')  # Save the current plot to the buffer as PNG
        buf.seek(0)
        frame_img = Image.open(buf) 
        frames.append(frame_img)  

    gif_buf = BytesIO()
    frames[0].save(gif_buf, save_all=True, append_images=frames[1:], duration=100, loop=0, format='GIF')

    gif_buf.seek(0)

    gif_base64 = base64.b64encode(gif_buf.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{gif_base64}" />'

# Helper function for detailed visualizations
def generate_visualizations(queue_system, simulation_time):
    # Queue Length Over Time
    plt.figure(figsize=(8, 4))
    time_steps = range(len(queue_system.queue_lengths))
    plt.plot(time_steps, queue_system.queue_lengths, label="Queue Length", color="orange")
    
    # Adding vertical lines for key events, e.g., high queue length moments
    max_queue_length = max(queue_system.queue_lengths)
    max_queue_time = queue_system.queue_lengths.index(max_queue_length)
    plt.axvline(x=max_queue_time, color='red', linestyle='--', label=f"Max Queue Length at {max_queue_time} mins")
    
    plt.title("Queue Length Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Queue Length")
    plt.grid()
    plt.legend()
    queue_length_plot = "queue_length_plot.png"
    plt.savefig(queue_length_plot)
    plt.close()

    # Waiting Times Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(queue_system.waiting_times, bins=20, kde=True, color="purple")
    
    # Add mean and median lines to the waiting times distribution plot
    mean_waiting_time = statistics.mean(queue_system.waiting_times)
    median_waiting_time = statistics.median(queue_system.waiting_times)
    plt.axvline(mean_waiting_time, color='blue', linestyle='--', label=f"Mean Waiting Time: {mean_waiting_time:.2f} mins")
    plt.axvline(median_waiting_time, color='green', linestyle='--', label=f"Median Waiting Time: {median_waiting_time:.2f} mins")
    
    plt.title("Distribution of Waiting Times")
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
    waiting_time_plot = "waiting_time_plot.png"
    plt.savefig(waiting_time_plot)
    plt.close()

    # Server Utilization
    plt.figure(figsize=(8, 4))
    utilization = [sum(queue_system.service_times) / (simulation_time * 60 * queue_system.num_servers) * 100]
    sns.barplot(x=["Server Utilization (%)"], y=utilization, palette="Blues_d")
    
    # Adding text annotations to show the server utilization percentage
    utilization_percentage = utilization[0]
    plt.text(0, utilization_percentage + 2, f"{utilization_percentage:.2f}%", ha="center", va="bottom", fontsize=12)
    
    plt.title("Server Utilization")
    plt.ylim(0, 100)
    plt.grid()
    utilization_plot = "utilization_plot.png"
    plt.savefig(utilization_plot)
    plt.close()

    return queue_length_plot, waiting_time_plot, utilization_plot

# Main app function
def queue_simulation(simulation_time, num_servers, service_rate, queue_discipline, logs: gr.State):
    logs.append("Starting simulation...")
    
    stats_summary = ""
    customer_data_table = ""
    queue_length_plot = ""
    waiting_time_plot = ""
    utilization_plot = ""
    queue_animation = ""
    
    # For "Both" discipline, run both FIFO and SJF simulations
    if queue_discipline == "Both":
        logs.append("Running FIFO simulation...")
        stats_fifo, customer_df_fifo, data_fifo, log_fifo = sm.run_simulation(
            simulation_time=simulation_time,
            num_servers=num_servers,
            service_rate=service_rate,
            queue_discipline="FIFO"
        )
        logs = logs + log_fifo
        logs.append("FIFO simulation completed.")
        
        logs.append("Running SJF simulation...")
        stats_sjf, customer_df_sjf, data_sjf, log_sjf = sm.run_simulation(
            simulation_time=simulation_time,
            num_servers=num_servers,
            service_rate=service_rate,
            queue_discipline="SJF"
        )
        logs = logs + log_sjf
        logs.append("SJF simulation completed.")
        
        # Generate visualizations for FIFO
        queue_system_fifo = QueueSystem(simpy.Environment(), num_servers, service_rate, "FIFO")
        queue_system_fifo.queue_lengths = data_fifo['queue_lengths']
        queue_system_fifo.waiting_times = data_fifo['waiting_times']
        queue_system_fifo.service_times = data_fifo['service_times']
        
        queue_length_plot_fifo, waiting_time_plot_fifo, utilization_plot_fifo = generate_visualizations(queue_system_fifo, simulation_time)
        
        # Generate visualizations for SJF
        queue_system_sjf = QueueSystem(simpy.Environment(), num_servers, service_rate, "SJF")
        queue_system_sjf.queue_lengths = data_sjf['queue_lengths']
        queue_system_sjf.waiting_times = data_sjf['waiting_times']
        queue_system_sjf.service_times = data_sjf['service_times']
        
        queue_length_plot_sjf, waiting_time_plot_sjf, utilization_plot_sjf = generate_visualizations(queue_system_sjf, simulation_time)
        
        # Generate animation for both FIFO and SJF
        queue_animation_fifo = create_queue_animation(queue_system_fifo.queue_lengths, simulation_time)
        queue_animation_sjf = create_queue_animation(queue_system_sjf.queue_lengths, simulation_time)
        
        # Create comparison plot for Queue Length
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(queue_system_fifo.queue_lengths)), queue_system_fifo.queue_lengths, label="FIFO", color="orange")
        plt.plot(range(len(queue_system_sjf.queue_lengths)), queue_system_sjf.queue_lengths, label="SJF", color="blue")
        plt.title("Queue Length Comparison (FIFO vs SJF)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Queue Length")
        plt.legend()
        comparison_plot = "comparison_plot.png"
        plt.savefig(comparison_plot)
        plt.close()
        
        stats_summary = (
            f"### Simulation Summary\n"
            f"- **Total Customers Served (FIFO)**: {stats_fifo['total_customers']}\n"
            f"- **Average Waiting Time (FIFO)**: {stats_fifo['average_waiting_time']:.2f} minutes\n"
            f"- **Average Queue Length (FIFO)**: {stats_fifo['average_queue_length']:.2f}\n"
            f"- **Average Time in System (FIFO)**: {stats_fifo['average_system_time']:.2f} minutes\n"
            f"- **Server Utilization (FIFO)**: {stats_fifo['server_utilization']:.2f}%\n"
            f"\n"
            f"- **Total Customers Served (SJF)**: {stats_sjf['total_customers']}\n"
            f"- **Average Waiting Time (SJF)**: {stats_sjf['average_waiting_time']:.2f} minutes\n"
            f"- **Average Queue Length (SJF)**: {stats_sjf['average_queue_length']:.2f}\n"
            f"- **Average Time in System (SJF)**: {stats_sjf['average_system_time']:.2f} minutes\n"
            f"- **Server Utilization (SJF)**: {stats_sjf['server_utilization']:.2f}%\n"
        )

        # Create customer data table
        customer_df = pd.concat([customer_df_fifo, customer_df_sjf], keys=["FIFO", "SJF"]).reset_index(level=0).rename(columns={'level_0': 'Discipline'})
        customer_data_table = customer_df.head(20).to_html(index=False)
        
    else:
        # Run simulation for selected discipline
        stats, customer_df, data, log = sm.run_simulation(
            simulation_time=simulation_time,
            num_servers=num_servers,
            service_rate=service_rate,
            queue_discipline=queue_discipline
        )
        
        logs = logs + log
        
        logs.append(f"{queue_discipline} simulation completed.")
        
        # Generate visualizations
        queue_system = QueueSystem(simpy.Environment(), num_servers, service_rate, queue_discipline)
        queue_system.queue_lengths = data['queue_lengths']
        queue_system.waiting_times = data['waiting_times']
        queue_system.service_times = data['service_times']

        queue_length_plot, waiting_time_plot, utilization_plot = generate_visualizations(queue_system, simulation_time)
        
        # Generate animation
        queue_animation = create_queue_animation(queue_system.queue_lengths, simulation_time)

        stats_summary = (
            f"### Simulation Summary\n"
            f"- **Total Customers Served**: {stats['total_customers']}\n"
            f"- **Average Waiting Time**: {stats['average_waiting_time']:.2f} minutes\n"
            f"- **Average Queue Length**: {stats['average_queue_length']:.2f}\n"
            f"- **Average Time in System**: {stats['average_system_time']:.2f} minutes\n"
            f"- **Server Utilization**: {stats['server_utilization']:.2f}%\n"
        )

    # Return appropriate outputs based on the discipline choice
    if queue_discipline == "Both":
        return stats_summary, customer_data_table, comparison_plot, waiting_time_plot_fifo, utilization_plot_fifo, queue_animation_fifo, "\n".join(logs)
    else:
        return stats_summary, customer_data_table, queue_length_plot, waiting_time_plot, utilization_plot, queue_animation, "\n".join(logs)

def list_files(queue_discipline, num_servers):
    if queue_discipline == "FIFO":
        directory = f"output/fifo"
        file_name = f"simulation_results_fifo_{num_servers}server.xlsx"
    elif queue_discipline == "SJF":
        directory = f"output/sjf"
        file_name = f"simulation_results_sjf_{num_servers}server.xlsx"
    else:
        return None
    
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        return file_path
    return None

def open_file(queue_discipline, num_servers):
    file_path = list_files(queue_discipline, num_servers)
    if file_path:
        # Open the Excel file and display its contents (first sheet)
        df = pd.read_excel(file_path, sheet_name=0)
        return df
    else:
        return f"No file found for {queue_discipline} with {num_servers} servers."


# UI layout
with gr.Blocks() as queue_sim_app:
    gr.Markdown("# 🚀 **Queue Simulation System for Performance Optimization**")
    gr.Markdown(
        """
        This interactive tool enables you to:
        - **Simulate Queueing Dynamics**: Analyze real-time behavior of customer queues under different configurations.
        - **Compare Queue Disciplines**: Understand and evaluate FIFO, SJF, or both to identify the best strategy for optimizing performance.
        - **Visualize Key Metrics**: Generate visualizations for waiting times, queue lengths, server utilization, etc.

        Tailor parameters to suit your scenario.
        """
    )
    
    server_options = list(range(2, 11))
    queue_options = ["FIFO", "SJF"]

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Simulation Parameters")
            simulation_time = gr.Slider(
                label="Simulation Time (hours)",
                minimum=1, maximum=24, step=1, value=12, interactive=True
            )
            num_servers = gr.Slider(
                label="Number of Servers",
                minimum=1, maximum=10, step=1, value=3, interactive=True
            )
            service_rate = gr.Slider(
                label="Service Rate (Mean Service Time)",
                minimum=5, maximum=50, step=5, value=10, interactive=True
            )
            queue_discipline = gr.Radio(
                label="Queue Discipline",
                choices=["FIFO", "SJF", "Both"],
                value="FIFO",
                interactive=True
            )
            simulate_button = gr.Button("🚀 Run Single Simulation")
            gr.Markdown("### Access Saved Results")
            with gr.Row():
                outputfile_discipline = gr.Dropdown(choices=queue_options, label="Queue Discipline (EXCEL)", value="FIFO")
                outputfile_numserver = gr.Dropdown(choices=server_options, label="Number of Servers (EXCEL)", value=2)
            with gr.Row():
                outputfile_view_button = gr.Button("🗂️ View Results")
                outputfile_open_button = gr.Button("📂 Open Results")
            outputfile_area = gr.Dataframe()
        with gr.Column():
            gr.Markdown("### Simulation Summary")
            stats_output = gr.Markdown()
            customer_data_output = gr.HTML()
            file_links_output = gr.HTML(
                label="Simulation Results Files")
            verbose_logs_output = gr.Textbox(
                label="Detailed Logs",
                lines=12, interactive=False, elem_id="logs_box")
        with gr.Column():
            gr.Markdown("### Visualizations")
            queue_length_plot_output = gr.Image(
                label="Queue Length Over Time")
            waiting_time_plot_output = gr.Image(
                label="Waiting Times Distribution")
            utilization_plot_output = gr.Image(
                label="Server Utilization")
            animation_output = gr.HTML()

    logs_state = gr.State([])
    
    def get_file_links(queue_discipline, num_servers):
        """Generate download links for saved simulation results."""
        if queue_discipline == "FIFO":
            file_path = os.path.join(sm.output_dir_fifo, f"simulation_results_fifo_{num_servers}server.xlsx")
        elif queue_discipline == "SJF":
            file_path = os.path.join(sm.output_dir_sjf, f"simulation_results_sjf_{num_servers}server.xlsx")
        else:
            return "<p>No files available for the selected discipline and configuration.</p>"

        if os.path.exists(file_path):
            return f'<a href="file://{file_path}" download>Download Simulation Results ({queue_discipline}, {num_servers} Servers)</a>'
        return "<p>No files found for the selected configuration.</p>"

    outputfile_open_button.click(
        lambda discipline, servers: get_file_links(discipline, servers),
        inputs=[outputfile_discipline, outputfile_numserver],
        outputs=file_links_output
    )
    
    outputfile_view_button.click(open_file, inputs=[outputfile_discipline, outputfile_numserver], outputs=outputfile_area)

    simulate_button.click(
        queue_simulation,
        inputs=[simulation_time, num_servers, service_rate, queue_discipline, logs_state],
        outputs=[
            stats_output,
            customer_data_output,
            queue_length_plot_output,
            waiting_time_plot_output,
            utilization_plot_output,
            animation_output,
            verbose_logs_output
        ]
    )

queue_sim_app.launch()
