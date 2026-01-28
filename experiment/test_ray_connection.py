"""
Test Ray Cluster Connectivity
==============================
Run this on MacBooks to test connection to Lenovo Ray head.

Usage on MacBook:
    python experiment/test_ray_connection.py 10.0.1.223
"""

import sys
import socket
import requests

def test_connectivity(head_ip: str):
    """Test connectivity to Ray head"""
    print("=" * 60)
    print("RAY CLUSTER CONNECTIVITY TEST")
    print("=" * 60)
    print(f"Head IP: {head_ip}")
    print()
    
    # Test 1: Ping ports
    ports = {
        6379: "Ray Head",
        8265: "Ray Dashboard",
        6380: "Ray GCS",
    }
    
    print("1. Testing port connectivity...")
    all_ok = True
    for port, name in ports.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((head_ip, port))
            sock.close()
            if result == 0:
                print(f"   ✅ Port {port} ({name}): OPEN")
            else:
                print(f"   ❌ Port {port} ({name}): CLOSED")
                all_ok = False
        except Exception as e:
            print(f"   ❌ Port {port} ({name}): ERROR - {e}")
            all_ok = False
    
    # Test 2: Dashboard HTTP
    print("\n2. Testing Ray Dashboard HTTP...")
    try:
        resp = requests.get(f"http://{head_ip}:8265/api/version", timeout=5)
        if resp.ok:
            print(f"   ✅ Dashboard API: OK - Ray {resp.json().get('version', 'unknown')}")
        else:
            print(f"   ❌ Dashboard API: HTTP {resp.status_code}")
            all_ok = False
    except Exception as e:
        print(f"   ❌ Dashboard API: {e}")
        all_ok = False
    
    # Test 3: Try Ray init
    print("\n3. Testing Ray connection...")
    try:
        import ray
        ray.init(address=f"ray://{head_ip}:6379", ignore_reinit_error=True)
        nodes = ray.nodes()
        alive = [n for n in nodes if n["Alive"]]
        print(f"   ✅ Ray connected: {len(alive)} nodes alive")
        
        # Print node info
        for node in alive:
            ip = node.get("NodeManagerAddress", "unknown")
            resources = node.get("Resources", {})
            cpus = resources.get("CPU", 0)
            gpus = resources.get("GPU", 0)
            print(f"      Node {ip}: {cpus} CPUs, {gpus} GPUs")
        
        ray.shutdown()
    except Exception as e:
        print(f"   ❌ Ray connection failed: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL TESTS PASSED - Ready to run distributed generation!")
        print()
        print("Next step - connect as worker:")
        print(f"   python experiment/10_ray_cluster_setup.py --worker --head-ip {head_ip}")
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Troubleshooting:")
        print("1. Check firewall on Lenovo:")
        print("   Run as Admin: .\\scripts\\open_ray_firewall.ps1")
        print("2. Make sure Ray head is running on Lenovo:")
        print("   python experiment/10_ray_cluster_setup.py --head")
        print("3. Both machines must be on same network")
        print(f"   Lenovo IP: {head_ip}")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        head_ip = "10.0.1.223"
        print(f"Using default head IP: {head_ip}")
    else:
        head_ip = sys.argv[1]
    
    test_connectivity(head_ip)
