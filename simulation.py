import simpy
import random
import statistics
import pandas as pd
import os

class QueueSystem:
    def __init__(self, env, num_servers, service_rate, queue_discipline="FIFO"):
        self.env = env
        self.server = simpy.PriorityResource(env, num_servers) if queue_discipline=="SJF" else simpy.Resource(env, num_servers)
        self.num_servers = num_servers
        self.service_rate = service_rate
        self.queue_discipline = queue_discipline

        # Statistics
        self.waiting_times = []
        self.service_times = []
        self.system_times = []
        self.queue_lengths = []
        self.customer_data = []

    def get_arrival_rate(self, hour):
        """
        Return arrival rate (customer per hour) based on time of day
        """
        if 0 <= hour < 7: # Early morning (low-demand)
            return 10
        elif 7 <= hour < 12: # Mid-morning (moderate demand)
            return 20
        elif 12 <= hour < 18: # Afternoon/evening (high demand)
            return 80
        elif 18 <= hour < 22: # Night peak (very high demand)
            return 160
        else: # Lowest demand outside of these time frames
            return 8

    def customer(self, customer_id):
        """
        Customer process for both FIFO and SJF disciplines
        """
        arrival_time = self.env.now

        self.queue_lengths.append(len(self.server.queue))

        # Estimate service time
        service_time = random.expovariate(1 / self.service_rate)
        self.service_times.append(service_time)

        if self.queue_discipline == "SJF":
            # PriorityResource: priority = service_time
            with self.server.request(priority=service_time) as request:
                yield request
                start_service_time = self.env.now

                # Calculate waiting time
                wait_time = max(0, start_service_time-arrival_time)
                self.waiting_times.append(wait_time)

                # Simulate service
                yield self.env.timeout(service_time)

                # Departure time
                departure_time = self.env.now

                # Track total system time
                system_time = departure_time - arrival_time
                self.system_times.append(system_time)

                self.customer_data.append({
                    "Customer ID": customer_id,
                    "Arrival Time": arrival_time,
                    "Wait Time": wait_time,
                    "Service Time": service_time,
                    "Departure Time": departure_time,
                    "System Time": system_time
                })

        elif self.queue_discipline == "FIFO":
            # FIFO is standard queue discipline for simpy
            with self.server.request() as request:
                yield request
                start_service_time = self.env.now

                # Calculate waiting time
                wait_time = max(0, start_service_time-arrival_time)
                self.waiting_times.append(wait_time)

                # Simulate service
                yield self.env.timeout(service_time)

                # Departure time
                departure_time = self.env.now

                # Track total system time
                system_time = departure_time - arrival_time
                self.system_times.append(system_time)

                self.customer_data.append({
                    "Customer ID": customer_id,
                    "Arrival Time": arrival_time,
                    "Wait Time": wait_time,
                    "Service Time": service_time,
                    "Departure Time": departure_time,
                    "System Time": system_time
                })

    def get_statistics(self, simulation_duration):
        """
        Calculate and return system statistics
        """
        avg_waiting_time = statistics.mean(self.waiting_times) if self.waiting_times else 0
        avg_queue_length = statistics.mean(self.queue_lengths) if self.queue_lengths else 0
        avg_system_time = statistics.mean(self.system_times) if self.system_times else 0
        total_busy_time = sum(self.service_times)
        server_utilization = (total_busy_time / (simulation_duration * self.num_servers)) * 100

        return {
            'average_waiting_time': avg_waiting_time,
            'average_queue_length': avg_queue_length,
            'average_system_time': avg_system_time,
            'server_utilization': server_utilization,
            'total_customers': len(self.waiting_times)
        }
        
logs = []

def run_simulation(simulation_time=24, num_servers=3, service_rate=10, queue_discipline="FIFO"):
    global logs
    
    """
    Run the queue simulation
    """
    env = simpy.Environment()
    queue_system = QueueSystem(env, num_servers, service_rate, queue_discipline)

    def customer_generator():
        customer_id = 0
        while True:
            current_hour = int(env.now // 60) % 24 # Determine hour of the day
            current_arrival_rate = queue_system.get_arrival_rate(current_hour)
            time_until_next = random.expovariate(current_arrival_rate / 60)
            yield env.timeout(time_until_next)
            env.process(queue_system.customer(customer_id))
            customer_id += 1

    # Start customer generator
    env.process(customer_generator())

    # Run simulation
    simulation_duration = simulation_time * 60 # Convert hours to minutes
    env.run(until=simulation_duration)

    # Generate and return statistics and customer data
    stats = queue_system.get_statistics(simulation_duration)
    customer_df = pd.DataFrame(queue_system.customer_data)
    
    data = {
        "queue_lengths": queue_system.queue_lengths,
        "waiting_times": queue_system.waiting_times,
        "service_times": queue_system.service_times,
        "system_times": queue_system.system_times,
        "customer_data": queue_system.customer_data
    }

    return stats, customer_df, data, logs


output_dir_fifo = os.path.join("output", "fifo")
output_dir_sjf = os.path.join("output", "sjf")
os.makedirs(output_dir_fifo, exist_ok=True)
os.makedirs(output_dir_sjf, exist_ok=True)
