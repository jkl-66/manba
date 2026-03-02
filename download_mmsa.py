import os
from MMSA import MMSA_Config

def download_datasets():
    data_dir = r"d:\code\论文\idea\Causal_MambaSA\data"
    os.makedirs(data_dir, exist_ok=True)
    
    # MMSA 内部集成了下载逻辑
    # 第一次运行特定任务时，它会自动检查并下载缺失的数据
    print("正在通过 MMSA 官方接口启动下载...")
    print("注意：这会下载包含 BERT 特征的对齐版 .pkl 文件。")
    
    # 这里的用法是利用 MMSA 的自动下载机制
    # 它会将数据下载到 ~/MMSA/datasets (默认路径)
    # 我们可以通过设置环境变量或手动移动来管理
    
    try:
        from MMSA.download import Downloader
        d = Downloader()
        
        # 下载 MOSEI
        print("\n--- 正在下载 CMU-MOSEI ---")
        d.download('mosei')
        
        # 下载 SIMS
        print("\n--- 正在下载 CH-SIMS ---")
        d.download('sims')
        
        print("\n下载完成！请前往默认路径 (通常是 ~/.mmsa/datasets 或当前目录下的 datasets) 查找文件。")
        print("然后将其移动到你的项目 data 文件夹中。")
        
    except Exception as e:
        print(f"自动下载失败: {e}")
        print("Plan C: 请访问以下 HuggingFace 直连地址手动下载。")

if __name__ == "__main__":
    download_datasets()
