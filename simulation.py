import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque
import seaborn as sns

# --- Task and Core Classes ---
class Task:
    def __init__(self, task_id, arrival_time, execution_time, priority):
        self.id = task_id
        self.arrival_time = arrival_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.priority = priority
        self.start_time = -1
        self.finish_time = -1
        self.assigned_core = -1
        self.is_completed = False
        self.last_run_time = -1

class Core:
    def __init__(self, core_id, threads_per_core):
        self.id = core_id
        self.threads_per_core = threads_per_core
        self.active_tasks = []
        self.busy_time = 0
        self.idle_time = 0
        self.l1_cache_hits = 0
        self.l1_cache_misses = 0

class SharedCache:
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def access(self):
        # Simulate L3 shared cache access (70% hit rate)
        if random.random() < 0.7:
            self.hits += 1
        else:
            self.misses += 1

# --- Task Generation ---
def generate_tasks(n=20, arrival_max=50, exec_min=2, exec_max=15, prio_min=1, prio_max=5):
    """Generate tasks with random arrival times, execution times, and priorities."""
    tasks = []
    for i in range(n):
        arrival = random.randint(0, arrival_max)
        duration = random.randint(exec_min, exec_max)
        priority = random.randint(prio_min, prio_max)
        tasks.append(Task(i, arrival, duration, priority))
    tasks.sort(key=lambda x: x.arrival_time)
    return tasks

# --- Scheduling Algorithms ---
def schedule_task(available_tasks, algorithm, current_time, rr_pointer=None):
    """Select a task based on the scheduling algorithm."""
    if not available_tasks:
        return None
    if algorithm == "RR":
        return available_tasks[rr_pointer % len(available_tasks)] if rr_pointer is not None else available_tasks[0]
    elif algorithm == "Priority":
        return min(available_tasks, key=lambda t: t.priority)
    elif algorithm == "SJF":
        return min(available_tasks, key=lambda t: t.remaining_time)
    return None

# --- Simulation Core ---
def run_simulation(
    num_cores, threads_per_core, algorithm, architecture="SMP", time_quantum=2, task_count=20, seed=42, num_threads=None
):
    """
    Simulate a multicore system with specified architecture and thread count.
    Models Intel Core i7-like architecture with 4 cores, 2 threads per core in Multicore configuration.
    """
    random.seed(seed)
    tasks = generate_tasks(task_count)
    cores = [Core(i, threads_per_core) for i in range(num_cores)]
    shared_cache = SharedCache()  # L3 shared cache
    context_switches = 0
    current_time = 0
    finished_tasks = []
    total_threads = num_threads if num_threads is not None else num_cores * threads_per_core
    gantt_log = []  # (task_id, core_id, start, finish)

    # Queue setup: shared for SMP/Multicore, per-core for Cluster
    if architecture == "Cluster":
        task_queues = [deque() for _ in range(num_cores)]
    else:
        task_queues = [deque()]

    next_task_idx = 0
    rr_pointers = [0 for _ in range(num_cores)]
    active_threads = 0

    while len(finished_tasks) < task_count:
        # Task arrival
        while next_task_idx < len(tasks) and tasks[next_task_idx].arrival_time <= current_time:
            task = tasks[next_task_idx]
            if architecture == "Cluster":
                core_id = random.randint(0, num_cores - 1)
                task_queues[core_id].append(task)
            else:
                task_queues[0].append(task)
            next_task_idx += 1

        # Assign tasks to cores
        for core_idx, core in enumerate(cores):
            available_slots = min(threads_per_core - len(core.active_tasks), total_threads - active_threads)
            for _ in range(available_slots):
                queue = task_queues[core_idx] if architecture == "Cluster" else task_queues[0]
                available = [t for t in queue if not t.is_completed and t.arrival_time <= current_time and t.remaining_time > 0]
                if not available:
                    continue
                rr_pointer = rr_pointers[core_idx] if algorithm == "RR" else None
                selected = schedule_task(available, algorithm, current_time, rr_pointer)
                if selected:
                    core.active_tasks.append(selected)
                    selected.assigned_core = core.id
                    if selected.start_time == -1:
                        selected.start_time = current_time
                    selected.last_run_time = current_time
                    queue.remove(selected)
                    context_switches += 1
                    gantt_log.append((selected.id, core.id, current_time, None))
                    active_threads += 1
                    if algorithm == "RR":
                        rr_pointers[core_idx] = (available.index(selected) + 1) % len(available) if available else 0

        # Advance simulation
        for core in cores:
            active_threads_current = len(core.active_tasks)
            core.busy_time += active_threads_current
            core.idle_time += (threads_per_core - active_threads_current)
            tasks_to_remove = []
            for task in core.active_tasks:
                # Simulate L1 cache access (80% hit rate)
                if random.random() < 0.8:
                    core.l1_cache_hits += 1
                else:
                    core.l1_cache_misses += 1
                    shared_cache.access()  # Miss triggers L3 access

                # RR preemption
                if algorithm == "RR" and (current_time - task.last_run_time) >= time_quantum:
                    if architecture == "Cluster":
                        task_queues[core.id].append(task)
                    else:
                        task_queues[0].append(task)
                    tasks_to_remove.append(task)
                    for entry in gantt_log[::-1]:
                        if entry[0] == task.id and entry[1] == core.id and entry[3] is None:
                            gantt_log[gantt_log.index(entry)] = (entry[0], entry[1], entry[2], current_time)
                            break
                    active_threads -= 1
                    continue
                # Execute task
                task.remaining_time -= 1
                if task.remaining_time <= 0:
                    task.finish_time = current_time + 1
                    task.is_completed = True
                    finished_tasks.append(task)
                    tasks_to_remove.append(task)
                    for entry in gantt_log[::-1]:
                        if entry[0] == task.id and entry[1] == core.id and entry[3] is None:
                            gantt_log[gantt_log.index(entry)] = (entry[0], entry[1], entry[2], current_time + 1)
                            break
                    active_threads -= 1
            for task in tasks_to_remove:
                core.active_tasks.remove(task)
                if not task.is_completed:
                    task.last_run_time = current_time + 1

        current_time += 1

    # Finalize Gantt log
    for i, entry in enumerate(gantt_log):
        if entry[3] is None:
            gantt_log[i] = (entry[0], entry[1], entry[2], current_time)

    # Metrics
    total_time = current_time
    cpu_util = sum(core.busy_time for core in cores) / (total_threads * total_time) * 100
    avg_turnaround = sum(t.finish_time - t.arrival_time for t in finished_tasks) / len(finished_tasks)
    throughput = len(finished_tasks) / total_time

    return {
        "architecture": architecture,
        "num_cores": num_cores,
        "threads_per_core": threads_per_core,
        "algorithm": algorithm,
        "cpu_utilization": cpu_util,
        "avg_turnaround": avg_turnaround,
        "throughput": throughput,
        "context_switches": context_switches,
        "total_time": total_time,
        "tasks": finished_tasks,
        "gantt_log": gantt_log,
        "l1_cache_hits": sum(core.l1_cache_hits for core in cores),
        "l1_cache_misses": sum(core.l1_cache_misses for core in cores),
        "l3_cache_hits": shared_cache.hits,
        "l3_cache_misses": shared_cache.misses
    }

# --- Gantt Chart ---
def plot_gantt_chart(gantt_log, num_cores, threads_per_core, algorithm, architecture, filename):
    """Plot a Gantt chart showing task execution on cores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for entry in gantt_log:
        task_id, core_id, start, finish = entry
        ax.broken_barh([(start, finish - start)], (core_id * 10, 9),
                       facecolors=colors[task_id % 20], edgecolors='black')
        ax.text(start + (finish - start) / 2, core_id * 10 + 4.5, f"T{task_id}",
                ha='center', va='center', color='white', fontsize=8)
    ax.set_ylim(0, num_cores * 10)
    ax.set_xlim(0, max(finish for _, _, _, finish in gantt_log) + 1)
    ax.set_xlabel('Time (units)')
    ax.set_yticks([i * 10 + 4.5 for i in range(num_cores)])
    ax.set_yticklabels([f'Core {i}' for i in range(num_cores)])
    ax.set_title(f'Gantt Chart: {algorithm} - {architecture} ({num_cores}C/{threads_per_core}T)')
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Excel Export ---
def save_results_to_excel(finished_tasks, algorithm, architecture, num_cores, threads_per_core, filename):
    """Save task metrics to Excel, including actual execution time."""
    df = pd.DataFrame([{
        "Task ID": t.id,
        "Arrival Time": t.arrival_time,
        "Execution Time": t.execution_time,
        "Actual Execution Time": t.finish_time - t.start_time if t.start_time != -1 and t.finish_time != -1 else 0,
        "Start Time": t.start_time,
        "Finish Time": t.finish_time,
        "Assigned Core": t.assigned_core,
        "Priority": t.priority
    } for t in finished_tasks])
    df.to_excel(filename, index=False)

# --- Thread Load Evaluation ---
def evaluate_thread_loads(
    num_cores=4, threads_per_core=2, algorithm="SJF", architecture="Multicore", time_quantum=2, task_count=20, seed=42
):
    """
    Evaluate the 4-core Multicore model with varying thread counts (1, 2, 4, 8).
    Simulates Intel Core i7-like architecture with Hyper-Threading (2 threads per core).
    """
    thread_counts = [1, 2, 4, 8]
    results = []
    for num_threads in thread_counts:
        print(f"Simulating: {architecture} | {num_cores}C/{threads_per_core}T | {num_threads} threads | {algorithm}")
        result = run_simulation(
            num_cores=num_cores,
            threads_per_core=threads_per_core,
            algorithm=algorithm,
            architecture=architecture,
            time_quantum=time_quantum,
            task_count=task_count,
            seed=seed,
            num_threads=num_threads
        )
        results.append({
            "architecture": architecture,
            "num_cores": num_cores,
            "threads_per_core": threads_per_core,
            "num_threads": num_threads,
            "algorithm": algorithm,
            "cpu_utilization": result["cpu_utilization"],
            "total_time": result["total_time"],
            "context_switches": result["context_switches"],
            "l1_cache_hits": result["l1_cache_hits"],
            "l1_cache_misses": result["l1_cache_misses"],
            "l3_cache_hits": result["l3_cache_hits"],
            "l3_cache_misses": result["l3_cache_misses"]
        })
        save_results_to_excel(
            result["tasks"], algorithm, architecture, num_cores, threads_per_core,
            f"results_{architecture}_{algorithm}_{num_cores}C_{num_threads}T.xlsx"
        )
        plot_gantt_chart(result["gantt_log"], num_cores, threads_per_core, algorithm, architecture,
                         f"gantt_{architecture}_{algorithm}_{num_cores}C_{num_threads}T.png")
    df = pd.DataFrame(results)
    df.to_excel(f"thread_load_comparison_{architecture}_{algorithm}.xlsx", index=False)
    print("\nThread Load Summary:")
    print(df)
    plot_thread_load_comparison(results, f"thread_load_{architecture}_{algorithm}.png")
    return results

# --- Comparative Plot ---
def plot_comparative_bar(results, metric, filename):
    """Plot a bar chart comparing architectures and algorithms."""
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="architecture", y=metric, hue="algorithm")
    plt.title(f'Comparison: {metric.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Thread Load Plot ---
def plot_thread_load_comparison(results, filename):
    """Plot execution time vs. thread count for the Multicore architecture."""
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="num_threads", y="total_time", hue="architecture")
    plt.title("Execution Time vs. Thread Count (Multicore, SJF)")
    plt.xlabel("Number of Threads")
    plt.ylabel("Total Execution Time (units)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main Evaluation ---
def evaluate_architectures():
    """
    Evaluate SMP, Multicore, and Cluster architectures with different scheduling algorithms.
    Multicore uses 4 cores, 2 threads per core to simulate Intel Core i7 (Nehalem) with Hyper-Threading.
    """
    configs = [
        ("SMP", 4, 1),  # 4 cores, no Hyper-Threading
        ("Multicore", 4, 2),  # 4 cores, 2 threads per core (Intel Core i7-like)
        ("Cluster", 4, 1)  # 4 nodes, distributed queues
    ]
    algorithms = ["SJF", "Priority", "RR"]
    all_results = []
    for arch, cores, threads in configs:
        for algo in algorithms:
            print(f"Simulating: {arch} | {cores}C/{threads}T | {algo}")
            result = run_simulation(
                num_cores=cores,
                threads_per_core=threads,
                algorithm=algo,
                architecture=arch,
                time_quantum=2,
                task_count=20,
                seed=42
            )
            all_results.append({
                "architecture": arch,
                "num_cores": cores,
                "threads_per_core": threads,
                "algorithm": algo,
                "cpu_utilization": result["cpu_utilization"],
                "avg_turnaround": result["avg_turnaround"],
                "throughput": result["throughput"],
                "context_switches": result["context_switches"],
                "l1_cache_hits": result["l1_cache_hits"],
                "l1_cache_misses": result["l1_cache_misses"],
                "l3_cache_hits": result["l3_cache_hits"],
                "l3_cache_misses": result["l3_cache_misses"]
            })
            plot_gantt_chart(result["gantt_log"], cores, threads, algo, arch,
                             f"gantt_{arch}_{algo}_{cores}C_{threads}T.png")
            save_results_to_excel(result["tasks"], algo, arch, cores, threads,
                                  f"results_{arch}_{algo}_{cores}C_{threads}T.xlsx")
        if arch == "Multicore":
            thread_results = evaluate_thread_loads(
                num_cores=cores,
                threads_per_core=threads,
                algorithm="SJF",
                architecture=arch,
                time_quantum=2,
                task_count=20,
                seed=42
            )
            all_results.extend(thread_results)
    print("\nSummary Table:")
    df = pd.DataFrame(all_results)
    print(df)
    for metric in ["cpu_utilization", "avg_turnaround", "throughput", "context_switches"]:
        plot_comparative_bar(all_results, metric, f"compare_{metric}.png")
    df.to_excel("architecture_comparison_summary.xlsx", index=False)
    print("All results exported.")

if __name__ == "__main__":
    try:
        evaluate_architectures()
    except Exception as e:
        print(f"Error during simulation: {e}")