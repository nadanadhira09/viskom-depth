"""
OPTIMIZATION PIPELINE - MASTER ORCHESTRATOR
=============================================
Menjalankan semua optimization steps secara berurutan:
1. GPU Setup verification
2. Batch inference optimization
3. ONNX graph optimization
4. Final competitive benchmark
5. Results summary & Android preparation
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("🚀 OPTIMIZATION PIPELINE - MASTER ORCHESTRATOR")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# PIPELINE STAGES
# ============================================================================
stages = [
    {
        "name": "GPU Setup Verification",
        "script": "gpu_setup_verification.py",
        "description": "Deteksi GPU dan install CUDA jika diperlukan"
    },
    {
        "name": "Batch Inference Optimization",
        "script": "batch_inference_optimization.py",
        "description": "Benchmark batch sizes 1, 5, 10 untuk optimal throughput"
    },
    {
        "name": "ONNX Graph Optimization",
        "script": "onnx_graph_optimization.py",
        "description": "Apply ORT_ENABLE_ALL dan thread optimization"
    },
    {
        "name": "Final Competitive Benchmark",
        "script": "final_competitive_benchmark.py",
        "description": "Benchmark lengkap dengan semua optimasi"
    },
]

# ============================================================================
# EXECUTION
# ============================================================================
results = []

for idx, stage in enumerate(stages, 1):
    stage_num = f"[{idx}/{len(stages)}]"
    
    print("\n" + "=" * 80)
    print(f"{stage_num} {stage['name'].upper()}")
    print("=" * 80)
    print(f"Description: {stage['description']}")
    print(f"Script: {stage['script']}")
    print(f"Status: Running...")
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run(
            [sys.executable, stage['script']],
            cwd=str(Path.cwd()),
            capture_output=False,
            timeout=600  # 10 minute timeout per stage
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            status = "✅ SUCCESS"
            results.append({
                "stage": stage['name'],
                "status": "SUCCESS",
                "time": elapsed
            })
        else:
            status = f"❌ ERROR (code {result.returncode})"
            results.append({
                "stage": stage['name'],
                "status": "ERROR",
                "time": elapsed
            })
        
        print(f"Status: {status} ({elapsed:.1f}s)")
        
    except subprocess.TimeoutExpired:
        print(f"Status: ⏱️ TIMEOUT (exceeded 600s)")
        results.append({
            "stage": stage['name'],
            "status": "TIMEOUT",
            "time": 600
        })
    except Exception as e:
        print(f"Status: ❌ EXCEPTION: {e}")
        results.append({
            "stage": stage['name'],
            "status": "EXCEPTION",
            "time": 0
        })

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 OPTIMIZATION PIPELINE - FINAL SUMMARY")
print("=" * 80)

total_time = sum(r['time'] for r in results)
success_count = sum(1 for r in results if r['status'] == 'SUCCESS')

print(f"\nExecution Summary:")
print(f"  • Total Stages: {len(results)}")
print(f"  • Successful: {success_count}")
print(f"  • Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")

print(f"\nDetailed Results:")
for idx, result in enumerate(results, 1):
    status_icon = {
        "SUCCESS": "✅",
        "ERROR": "❌",
        "TIMEOUT": "⏱️",
        "EXCEPTION": "💥"
    }.get(result['status'], "❓")
    
    print(f"  {idx}. {status_icon} {result['stage']:40s} ({result['time']:.1f}s)")

# ============================================================================
# NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("🎯 NEXT STEPS - ANDROID DEPLOYMENT")
print("=" * 80)

if success_count == len(results):
    print("""
✅ ALL OPTIMIZATION STAGES COMPLETE!

Next: Prepare for Android Implementation
1. Review optimization_results.txt
2. Create android_deployment_config.py with best settings
3. Copy Kotlin templates to Android Studio project
4. Build and deploy to device

Commands:
  python android_deployment_config.py  # Generate optimal settings
  # Then open Android Studio and follow QUICK_START.md
""")
else:
    print(f"""
⚠️  {len(results) - success_count} stage(s) failed or timed out.

Review the error messages above and run individual scripts:
  • python gpu_setup_verification.py
  • python batch_inference_optimization.py
  • python onnx_graph_optimization.py
  • python final_competitive_benchmark.py
""")

print("\n" + "=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
