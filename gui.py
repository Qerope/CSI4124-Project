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

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def create_queue_animation(all_queue_lengths, average_queue_length, num_servers, label):
    fig, ax = plt.subplots(figsize=(8, 4))
    max_length = max(max(len(run) for run in all_queue_lengths), len(average_queue_length))
    max_height = max(max(max(run) for run in all_queue_lengths), max(average_queue_length)) + 5
    
    ax.set_xlim(0, max_length)
    ax.set_ylim(0, max_height)
    ax.set_title(f"Queue Length Per Customer {label} m={num_servers}")
    ax.set_xlabel("Customer")
    ax.set_ylabel("Queue Length")
    lines = [ax.plot([], [], lw=1, alpha=0.6)[0] for _ in all_queue_lengths]
    avg_line, = ax.plot([], [], lw=2, color="red", label="Average")
    ax.legend()

    frames = []

    for frame in range(max_length):
        for line, queue_lengths in zip(lines, all_queue_lengths):
            line.set_data(range(min(frame, len(queue_lengths))), queue_lengths[:frame])
        avg_line.set_data(range(min(frame, len(average_queue_length))), average_queue_length[:frame])
        
        buf = BytesIO()
        plt.savefig(buf, format='png')  # Save current plot to buffer as PNG
        buf.seek(0)
        frame_img = Image.open(buf)
        frames.append(frame_img)

    gif_buf = BytesIO()
    frames[0].save(gif_buf, save_all=True, append_images=frames[1:], duration=100, loop=0, format='GIF')

    gif_buf.seek(0)

    filename = os.path.join(data_dir, f"animated_queue_lengths_{label}_m={num_servers}_{num_servers}.gif")

    gif_base64 = base64.b64encode(gif_buf.read()).decode('utf-8')
    with open(filename, "wb") as f:
        f.write(base64.b64decode(gif_base64))
    return filename

# Helper function for detailed visualizations
def generate_visualizations(all_queue_lengths, average_queue_length, all_waiting_times, average_waiting_time, simulation_time, all_service_times, num_servers, label):
    # Queue Length Over Time
    plt.figure(figsize=(8, 4))
    for queue_lengths in all_queue_lengths:
        plt.plot(range(len(queue_lengths)), queue_lengths, alpha=0.5, color="lightblue")
    plt.plot(range(len(average_queue_length)), average_queue_length, label="Average", color="red")
    plt.title(f"Queue Length Over Time {label} m={num_servers}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Queue Length")
    plt.grid()
    plt.legend()
    queue_length_plot = os.path.join(data_dir, f"queue_length_plot_{label}_m{num_servers}.png")
    plt.savefig(queue_length_plot)
    plt.close()

    # Waiting Times Distribution
    plt.figure(figsize=(8, 4))
    for waiting_times in all_waiting_times:
        sns.histplot(waiting_times, bins=20, kde=True, alpha=0.5, color="lightblue")
    sns.histplot(average_waiting_time, bins=20, kde=True, color="red", label="Average")
    plt.title(f"Distribution of Waiting Times {label} m={num_servers}")
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Frequency")
    plt.legend()
    waiting_time_plot = os.path.join(data_dir, f"waiting_time_plot_{label}_m{num_servers}.png")
    plt.savefig(waiting_time_plot)
    plt.close()
    
    
    # Server Utilization
    plt.figure(figsize=(8, 4))

    utilizations = []
    for service_time in all_service_times:
        utilization = sum(service_time) / (simulation_time * 60 * int(num_servers)) * 100
        utilizations.append(utilization)

    sns.barplot(x=[f"{i+1}" for i in range(len(utilizations))], y=utilizations, alpha=0.6, color="lightblue")

    average_utilization = sum(utilizations) / len(utilizations)
    sns.barplot(x=["AVG"], y=[average_utilization], color="red", alpha=0.8)

    utilization_percentage = utilization
    plt.text(0, 0, f"{utilization_percentage:.2f}%", ha="center", va="bottom", fontsize=12)
    
    plt.title(f"Server Utilization {label} m={num_servers}")
    plt.ylim(0, 100)
    plt.grid()
    utilization_plot = os.path.join(data_dir, f"utilization_plot_{label}_m{num_servers}.png")
    plt.savefig(utilization_plot)
    plt.close()

    # Server Utilization
    # Add a bar plot with averages
    return queue_length_plot, waiting_time_plot, utilization_plot

# Main app function
def queue_simulation(num_runs, simulation_time, num_servers, service_rate, queue_discipline, logs):
    logs.append("Starting simulation...")

    stats_summary = ""
    customer_data_table = ""
    queue_length_plot = ""
    waiting_time_plot = ""
    utilization_plot = ""
    queue_animation = ""

    all_queue_lengths = []
    all_waiting_times = []
    all_utilizations = []
    all_queue_lengths_fifo = []
    all_waiting_times_fifo = []
    all_utilizations_fifo = []
    all_queue_lengths_sjf = []
    all_waiting_times_sjf = []
    all_utilizations_sjf = []

    all_waiting_times_fifo_list = []
    all_queue_lengths_fifo_list = []
    all_system_time_fifo_list = []
    all_waiting_times_sjf_list = []
    all_queue_lengths_sjf_list = []
    all_system_time_sjf_list = []
    all_waiting_times_list = []
    all_queue_lengths_list = []
    all_system_time_list = []
    all_service_time_fifo_list = []
    all_service_time_sjf_list = []
    all_service_time_list = []

    # Run simulations for both FIFO and SJF if "Both" discipline is selected
    if queue_discipline == "Both":
        logs.append("Running FIFO simulations...")

        for run in range(num_runs):
            stats_fifo, customer_df_fifo, data_fifo, log_fifo = sm.run_simulation(
                simulation_time=simulation_time,
                num_servers=num_servers,
                service_rate=service_rate,
                queue_discipline="FIFO"
            )
            logs.append(f"Run {run + 1} for FIFO completed.")
            all_queue_lengths_fifo.append(data_fifo['queue_lengths'])
            all_waiting_times_fifo.append(data_fifo['waiting_times'])
            all_utilizations_fifo.append(stats_fifo['server_utilization'])
            
            all_waiting_times_fifo_list.append(stats_fifo['average_waiting_time'])
            all_queue_lengths_fifo_list.append(stats_fifo['average_queue_length'])
            all_system_time_fifo_list.append(stats_fifo['average_system_time'])
            all_service_time_fifo_list.append(data_fifo["service_times"])

        logs.append("FIFO simulation completed.")
        
        logs.append("Running SJF simulations...")

        for run in range(num_runs):
            stats_sjf, customer_df_sjf, data_sjf, log_sjf = sm.run_simulation(
                simulation_time=simulation_time,
                num_servers=num_servers,
                service_rate=service_rate,
                queue_discipline="SJF"
            )
            logs.append(f"Run {run + 1} for SJF completed.")
            all_queue_lengths_sjf.append(data_sjf['queue_lengths'])
            all_waiting_times_sjf.append(data_sjf['waiting_times'])
            all_utilizations_sjf.append(stats_sjf['server_utilization'])
            
            all_waiting_times_sjf_list.append(stats_sjf['average_waiting_time'])
            all_queue_lengths_sjf_list.append(stats_sjf['average_queue_length'])
            all_system_time_sjf_list.append(stats_sjf['average_system_time'])
            all_service_time_sjf_list.append(data_sjf["service_times"])

        logs.append("SJF simulation completed.")

        # Calculate averages for FIFO and SJF (flatten the lists and then calculate averages)
        average_queue_length_fifo = [sum(x) / len(x) for x in zip(*all_queue_lengths_fifo)]
        average_waiting_time_fifo = [sum(x) / len(x) for x in zip(*all_waiting_times_fifo)]
        
        average_queue_length_sjf = [sum(x) / len(x) for x in zip(*all_queue_lengths_sjf)]
        average_waiting_time_sjf = [sum(x) / len(x) for x in zip(*all_waiting_times_sjf)]

        # Generate visualizations for FIFO
        queue_length_plot_fifo, waiting_time_plot_fifo, utilization_plot_fifo = generate_visualizations(
            all_queue_lengths_fifo, average_queue_length_fifo, all_waiting_times_fifo, average_waiting_time_fifo, simulation_time, all_service_time_fifo_list, num_servers, "FIFO"
        )
        
        # Generate visualizations for SJF
        queue_length_plot_sjf, waiting_time_plot_sjf, utilization_plot_sjf = generate_visualizations(
            all_queue_lengths_sjf, average_queue_length_sjf, all_waiting_times_sjf, average_waiting_time_sjf, simulation_time, all_service_time_sjf_list, num_servers, "SJF"
        )

        # Generate animation for both FIFO and SJF
        queue_animation_fifo = create_queue_animation(all_queue_lengths_fifo, average_queue_length_fifo, num_servers, "FIFO")
        queue_animation_sjf = create_queue_animation(all_queue_lengths_sjf, average_queue_length_sjf, num_servers, "SJF")

        # Create comparison plot for Queue Length
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(average_queue_length_fifo)), average_queue_length_fifo, label="FIFO Average", color="orange")
        plt.plot(range(len(average_queue_length_sjf)), average_queue_length_sjf, label="SJF Average", color="blue")
        plt.title(f"Queue Length Comparison (FIFO vs SJF) m={num_servers}")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Queue Length")
        plt.legend()
        comparison_plot = os.path.join(data_dir, f"comparison_plot_{num_servers}.png")
        plt.savefig(comparison_plot)
        plt.close()

        stats_summary = (
            f"### m={num_servers} (Averaged over {num_runs} runs)\n"
            f"- **Average Server Utilization (FIFO)**: {statistics.mean(all_utilizations_fifo):.2f}%\n"
            f"- **Average Server Utilization (SJF)**: {statistics.mean(all_utilizations_sjf):.2f}%\n"
            f"\n"
            f"- **Average Waiting Time (FIFO)**: {statistics.mean(all_waiting_times_fifo_list):.2f} minutes\n"
            f"- **Average Waiting Time (SJF)**: {statistics.mean(all_waiting_times_sjf_list):.2f} minutes\n"
            f"- **Average Queue Length (FIFO)**: {statistics.mean(all_queue_lengths_fifo_list):.2f}\n"
            f"- **Average Queue Length (SJF)**: {statistics.mean(all_queue_lengths_sjf_list):.2f}\n"
            f"- **Average Time in System (FIFO)**: {statistics.mean(all_system_time_fifo_list):.2f} minutes\n"
            f"- **Average Time in System (SJF)**: {statistics.mean(all_system_time_sjf_list):.2f} minutes\n"
        )

        # Create customer data table
        customer_df = pd.concat([customer_df_fifo, customer_df_sjf], keys=["FIFO", "SJF"]).reset_index(level=0).rename(columns={'level_0': 'Discipline'})
        customer_data_table = customer_df.round(2).to_html(index=False)

        # Return all outputs
        return stats_summary, customer_data_table, comparison_plot, waiting_time_plot_fifo, waiting_time_plot_sjf, utilization_plot_fifo, utilization_plot_sjf, queue_animation_fifo, queue_animation_sjf, "\n".join(logs)
    
    else:
        # Run for the selected discipline
        for run in range(num_runs):
            stats, customer_df, data, log = sm.run_simulation(
                simulation_time=simulation_time,
                num_servers=num_servers,
                service_rate=service_rate,
                queue_discipline=queue_discipline
            )
            logs.append(f"Run {run + 1} for {queue_discipline} completed.")
            all_queue_lengths.append(data['queue_lengths'])
            all_waiting_times.append(data['waiting_times'])
            all_utilizations.append(stats['server_utilization'])
            
            all_waiting_times_list.append(stats['average_waiting_time'])
            all_queue_lengths_list.append(stats['average_queue_length'])
            all_system_time_list.append(stats['average_system_time'])
            all_service_time_list.append(data["service_times"])


        logs.append(f"{queue_discipline} simulation completed.")

        # Calculate averages for single run
        average_queue_length = [sum(x) / len(x) for x in zip(*all_queue_lengths)]
        average_waiting_time = [sum(x) / len(x) for x in zip(*all_waiting_times)]

        # Generate visualizations
        queue_length_plot, waiting_time_plot, utilization_plot = generate_visualizations(
            all_queue_lengths, average_queue_length, all_waiting_times, average_waiting_time, simulation_time, all_service_time_list, num_servers, queue_discipline
        )
        
        customer_data_table = customer_df.round(2).to_html(index=False)

        # Generate animation
        queue_animation = create_queue_animation(all_queue_lengths, average_queue_length, num_servers, queue_discipline)

        stats_summary = (
            f"### m={num_servers} (Averaged over {num_runs} runs)\n"
            f"- **Average Server Utilization**: {statistics.mean(all_utilizations):.2f}%\n"
            f"- **Average Waiting Time**: {statistics.mean(all_waiting_times_list):.2f} minutes\n"
            f"- **Average Queue Length**: {statistics.mean(all_queue_lengths_list):.2f}\n"
            f"- **Average Time in System**: {statistics.mean(all_system_time_list):.2f}\n"
        )

        # single discipline results
        return stats_summary, customer_data_table, queue_length_plot, waiting_time_plot, waiting_time_plot, utilization_plot, utilization_plot, queue_animation, queue_animation, "\n".join(logs)

def list_files(queue_discipline, num_servers):
    if queue_discipline == "FIFO":
        directory = f"output/fifo"
        file_name = f"simulation_results_{num_servers}server.xlsx"
    elif queue_discipline == "SJF":
        directory = f"output/sjf"
        file_name = f"simulation_results_{num_servers}server.xlsx"
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
        ### Please run the simulation and scroll down for visualizations

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
            num_runs = gr.Slider(
                label="Number of Runs (N)",
                minimum=1, maximum=70, step=1, value=1, interactive=True
            )
            simulation_time = gr.Slider(
                label="Simulation Time (hours)",
                minimum=1, maximum=24, step=1, value=12, interactive=True
            )
            with gr.Row():
                num_servers_min = gr.Slider(
                    label="Minimum Number of Servers (m_min)",
                    minimum=1, maximum=12, step=1, value=2, interactive=True
                )
                num_servers_max = gr.Slider(
                    label="Maximum Number of Servers (m_min)",
                    minimum=1, maximum=12, step=1, value=10, interactive=True
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
            simulate_button = gr.Button("🚀 Run Simulation")
            verbose_logs_output = gr.Textbox(
                label="Logs",
                lines=18, interactive=False, elem_id="logs_box")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Simulation Summary #1")
                    stats_output1 = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Simulation Summary #2")
                    stats_output2 = gr.Markdown()
            gr.Markdown("### Access Saved Results (EXCEL DATA) -> 30 Runs FIFO vs SJF m=[2,10]")
            run_defined_simulations = gr.Button("Run Predefined Simulations")
            file_links_output = gr.HTML(
                label="Simulation Results Files")
            with gr.Row():
                outputfile_discipline = gr.Dropdown(choices=queue_options, label="Queue Discipline", value="FIFO")
                outputfile_numserver = gr.Dropdown(choices=server_options, label="Number of Servers", value=2)
            with gr.Row():
                outputfile_view_button = gr.Button("🗂️ View Results")
                outputfile_open_button = gr.Button("📂 Open Results")
            outputfile_area = gr.Dataframe()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Visualizations")
            with gr.Row():
                with gr.Column():
                    queue_length_plot_output = gr.Gallery(
                        label="Queue Length Over Time")
                    waiting_time_plot_output = gr.Gallery(
                        label="Waiting Times Distribution #1")
                    waiting_time_plot_output2 = gr.Gallery(
                        label="Waiting Times Distribution #2")
                with gr.Column():
                    utilization_plot_output = gr.Gallery(
                        label="Server Utilization #1")
                    utilization_plot_output2 = gr.Gallery(
                        label="Server Utilization #2")
                    animation_output = gr.Gallery(
                        label="Queue Length #1")
                    animation_output2 = gr.Gallery(
                        label="Queue Length #2")
    with gr.Row():
        customer_data_output = gr.HTML()

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


    def on_button_click():
        result = sm.run_simulations([])  # Run the simulations
        return result
    
    outputfile_view_button.click(open_file, inputs=[outputfile_discipline, outputfile_numserver], outputs=outputfile_area)

    run_defined_simulations.click(on_button_click, outputs=file_links_output)

    verbose_logs = []
    verbose_logs_str = ""
    stats = ""

    def simulate(queue_discipline, num_servers_min, num_servers_max, num_runs, simulation_time, service_rate):
        global verbose_logs, verbose_logs_str, stats
        stats = ""
        gallery_data_queue_length = []
        gallery_data_waiting_times = []
        gallery_data_waiting_times2 = []
        gallery_data_utilization = []
        gallery_data_utilization2 = []
        gallery_animation_output = []
        gallery_animation_output2 = []
        
        for num_servers in range(num_servers_min, num_servers_max + 1):
            stats_n, customer_data, queue_length_plot, waiting_time_plot, waiting_time_plot2, utilization_plot, utilization_plot2, animation, animation2, verbose_logs_n = queue_simulation( num_runs, simulation_time, num_servers, service_rate, queue_discipline, [] )

            verbose_logs = [verbose_logs_str] + [verbose_logs_n]
            verbose_logs_str = '\n'.join(verbose_logs)
            stats = stats + stats_n
            
            # Append to galleries
            gallery_data_queue_length.append(queue_length_plot)
            gallery_data_waiting_times.append(waiting_time_plot)
            gallery_data_utilization.append(utilization_plot)
            gallery_data_waiting_times2.append(waiting_time_plot2)
            gallery_data_utilization2.append(utilization_plot2)
            gallery_animation_output.append(animation)
            gallery_animation_output2.append(animation2)
        
        stat_part1, stat_part2 = "\n".join(stats.splitlines()[:len(stats.splitlines())//2]), "\n".join(stats.splitlines()[len(stats.splitlines())//2:])

        return stat_part1, stat_part2, customer_data, gallery_data_queue_length, gallery_data_waiting_times, gallery_data_waiting_times2, gallery_data_utilization, gallery_data_utilization2, gallery_animation_output, gallery_animation_output2, verbose_logs_str

    simulate_button.click(
        simulate,
        inputs=[queue_discipline, num_servers_min, num_servers_max, num_runs, simulation_time, service_rate],
        outputs=[
            stats_output1,
            stats_output2,
            customer_data_output,
            queue_length_plot_output,
            waiting_time_plot_output,
            waiting_time_plot_output2,
            utilization_plot_output,
            utilization_plot_output2,
            animation_output,
            animation_output2,
            verbose_logs_output
        ]
    )

queue_sim_app.launch()
