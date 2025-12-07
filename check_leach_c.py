"""
Script kiểm tra nhanh LEACH-C hoạt động đúng
"""

from env.wsn_env import WSNEnvironment
import numpy as np

print("=" * 60)
print("🧪 KIỂM TRA LEACH-C")
print("=" * 60)

env = WSNEnvironment(n_nodes=30, area=100)
env.reset()

print(f"\n📍 Thông tin mạng:")
print(f"   - Số nút: {env.n}")
print(f"   - Vị trí BS: {env.bs}")
print(f"   - Năng lượng ban đầu: {env.initial_energy} J")

print(f"\n⚙️ Thiết lập LEACH-C với 3 clusters...")
env.setup_leach_c(num_clusters=3)

print(f"\n👑 Cluster Heads được chọn: {env.leaders}")
print(f"\n📊 Cấu trúc clusters:")
for i, cluster in enumerate(env.clusters):
    print(f"   Cluster {i} (CH={env.leaders[i]}): {len(cluster)} nút")

print(f"\n📡 Mô phỏng 1 round truyền dữ liệu...")
reward, done, info = env.transmit_data_leach_c()

print(f"\n📈 Kết quả:")
print(f"   - Reward: {reward:.2f}")
print(f"   - Packets: {info['packets']}")
print(f"   - Alive ratio: {info['alive_ratio']:.2%}")

print(f"\n🔄 Chạy thêm 10 rounds...")
for r in range(10):
    env.setup_leach_c(num_clusters=3)
    reward, done, info = env.transmit_data_leach_c()
    print(f"   Round {r+2}: Alive={np.sum(env.alive)}, Packets={info['packets']}")
    if done:
        break

print("\n" + "=" * 60)
print("✅ KIỂM TRA HOÀN TẤT!")
print("=" * 60)
