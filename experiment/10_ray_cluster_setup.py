"""
Step 10: Ray Cluster Setup for Distributed Computing
=====================================================
Sets up Ray cluster across 3 machines for faster data generation:
- Lenovo LOQ (RTX 4060) - HEAD NODE + GPU worker
- MacBook Pro - CPU worker
- MacBook Air - CPU worker

Usage:
    # On HEAD node (Lenovo LOQ):
    python experiment/10_ray_cluster_setup.py --head --port 6379
    
    # On worker nodes (MacBooks):
    python experiment/10_ray_cluster_setup.py --worker --head-ip <LENOVO_IP>
    
    # Check cluster status:
    python experiment/10_ray_cluster_setup.py --status
"""

import os
import sys
import argparse
import socket
import subprocess
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default Ray port
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_system_info():
    """Get system information for resource allocation"""
    import psutil
    
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "ip": get_local_ip(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "has_gpu": False,
        "gpu_name": None
    }
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    except ImportError:
        pass
    
    return info


def start_head_node(port: int = RAY_PORT):
    """Start Ray head node"""
    import ray
    
    info = get_system_info()
    print("=" * 60)
    print("STARTING RAY HEAD NODE")
    print("=" * 60)
    print(f"Hostname: {info['hostname']}")
    print(f"IP Address: {info['ip']}")
    print(f"Platform: {info['platform']}")
    print(f"CPUs: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB")
    if info["has_gpu"]:
        print(f"GPU: {info['gpu_name']} ({info.get('gpu_memory_gb', 'N/A')} GB)")
    print("=" * 60)
    
    # Configure resources based on system
    num_cpus = info["cpu_count"] - 1  # Reserve 1 for system
    num_gpus = 1 if info["has_gpu"] else 0
    
    # Start Ray head
    ray.init(
        address=None,  # Start new cluster
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        dashboard_host="0.0.0.0",
        dashboard_port=RAY_DASHBOARD_PORT,
        include_dashboard=True,
        _temp_dir=str(PROJECT_ROOT / "ray_temp"),
    )
    
    print(f"\n✅ Ray head node started!")
    print(f"\n📋 WORKER CONNECTION INFO:")
    print(f"   Head IP: {info['ip']}")
    print(f"   Ray Address: ray://{info['ip']}:{port}")
    print(f"   Dashboard: http://{info['ip']}:{RAY_DASHBOARD_PORT}")
    print(f"\n📝 To connect workers, run on each MacBook:")
    print(f"   python experiment/10_ray_cluster_setup.py --worker --head-ip {info['ip']}")
    print(f"\n⏳ Waiting for workers... (Press Ctrl+C to stop)")
    
    # Keep running and show cluster status
    try:
        import time
        while True:
            time.sleep(10)
            nodes = ray.nodes()
            alive = [n for n in nodes if n["Alive"]]
            print(f"\r🖥️  Connected nodes: {len(alive)}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Ray cluster...")
        ray.shutdown()


def connect_worker(head_ip: str, port: int = RAY_PORT):
    """Connect to Ray cluster as worker"""
    import ray
    
    info = get_system_info()
    print("=" * 60)
    print("CONNECTING AS RAY WORKER")
    print("=" * 60)
    print(f"Hostname: {info['hostname']}")
    print(f"Platform: {info['platform']}")
    print(f"CPUs: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB")
    print(f"Connecting to: ray://{head_ip}:{port}")
    print("=" * 60)
    
    # Connect to existing cluster
    ray.init(address=f"ray://{head_ip}:{port}")
    
    print(f"\n✅ Connected to Ray cluster!")
    print(f"   This node contributes: {info['cpu_count']} CPUs")
    
    # Keep connection alive
    try:
        import time
        while True:
            time.sleep(30)
            print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Disconnecting from cluster...")
        ray.shutdown()


def show_cluster_status(head_ip: str = None, port: int = RAY_PORT):
    """Show Ray cluster status"""
    import ray
    
    if head_ip:
        ray.init(address=f"ray://{head_ip}:{port}", ignore_reinit_error=True)
    else:
        ray.init(address="auto", ignore_reinit_error=True)
    
    print("=" * 60)
    print("RAY CLUSTER STATUS")
    print("=" * 60)
    
    nodes = ray.nodes()
    total_cpus = 0
    total_gpus = 0
    total_memory = 0
    
    for i, node in enumerate(nodes):
        status = "✅ ALIVE" if node["Alive"] else "❌ DEAD"
        resources = node.get("Resources", {})
        cpus = resources.get("CPU", 0)
        gpus = resources.get("GPU", 0)
        memory = resources.get("memory", 0) / (1024**3)
        
        print(f"\nNode {i + 1}: {status}")
        print(f"  Address: {node.get('NodeManagerAddress', 'N/A')}")
        print(f"  CPUs: {cpus}")
        print(f"  GPUs: {gpus}")
        print(f"  Memory: {memory:.1f} GB")
        
        if node["Alive"]:
            total_cpus += cpus
            total_gpus += gpus
            total_memory += memory
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL CLUSTER RESOURCES:")
    print(f"  CPUs: {total_cpus}")
    print(f"  GPUs: {total_gpus}")
    print(f"  Memory: {total_memory:.1f} GB")
    print(f"{'=' * 60}")
    
    ray.shutdown()


def install_ray():
    """Install Ray if not present"""
    try:
        import ray
        print(f"✅ Ray already installed: {ray.__version__}")
    except ImportError:
        print("📦 Installing Ray...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ray[default]"], check=True)
        print("✅ Ray installed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Ray Cluster Setup")
    parser.add_argument("--head", action="store_true", help="Start as head node")
    parser.add_argument("--worker", action="store_true", help="Connect as worker node")
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    parser.add_argument("--head-ip", type=str, help="Head node IP address")
    parser.add_argument("--port", type=int, default=RAY_PORT, help="Ray port")
    parser.add_argument("--install", action="store_true", help="Install Ray")
    
    args = parser.parse_args()
    
    if args.install:
        install_ray()
        return
    
    if args.head:
        start_head_node(args.port)
    elif args.worker:
        if not args.head_ip:
            print("❌ Error: --head-ip required for worker mode")
            sys.exit(1)
        connect_worker(args.head_ip, args.port)
    elif args.status:
        show_cluster_status(args.head_ip, args.port)
    else:
        # Just show system info
        info = get_system_info()
        print("\n📊 System Information:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("\nUsage:")
        print("  Head node:   python 10_ray_cluster_setup.py --head")
        print("  Worker node: python 10_ray_cluster_setup.py --worker --head-ip <IP>")
        print("  Status:      python 10_ray_cluster_setup.py --status")


if __name__ == "__main__":
    main()
