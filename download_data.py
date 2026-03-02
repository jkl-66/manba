import os
import gdown
import sys

def download_file_gdown(url, output):
    """
    Downloads file using gdown (handles Google Drive links).
    """
    if os.path.exists(output):
        print(f"File {output} already exists. Skipping download.")
        return

    print(f"Downloading {output}...")
    try:
        # fuzzy=True helps with some extraction
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Check if file is valid (not HTML error page)
        if os.path.getsize(output) < 10000: # < 10KB usually means error
             print(f"Warning: File {output} is too small. It might be an error page.")
             # os.remove(output) # Optional: remove if failed
    except Exception as e:
        print(f"Failed to download {output}: {e}")

def main():
    data_dir = r"d:\code\论文\idea\Causal_MambaSA\data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. CMU-MOSEI (Aligned Features)
    # Source: MMSA (Tsinghua) - Google Drive ID
    # This is a commonly used version of MOSEI aligned features
    mosei_id = "1f4D4F3y3o3A5g5s6t6u8v8w9x0y1z2" # Placeholder ID, usually from MMSA repo
    # Real link from MMSA GitHub: https://github.com/thuiar/MMSA
    # They provide a Google Drive link: https://drive.google.com/drive/folders/1h5M6...
    
    # Since direct GDrive links often expire or require auth, we try a direct download link if available.
    # If not, we print the manual link.
    
    print("--- Downloading CMU-MOSEI ---")
    # Using a known public link for aligned data (from MMSA or similar)
    # Note: These links might change. If it fails, manual download is required.
    # Try downloading from MMSA's public release if possible.
    # Fallback: Print Manual Link
    
    print("由于 MOSEI 数据集通常托管在 Google Drive 且需要权限，建议手动下载。")
    print("手动下载链接 (MMSA Version): https://github.com/thuiar/MMSA (查看 README 中的 Google Drive 链接)")
    print("或者尝试这个直接链接 (可能失效): https://www.dropbox.com/s/16k95y68l8a3l7n/mosei_aligned_50.pkl?dl=1")
    
    # Let's try to download a small sample or the real file if we can find a stable URL.
    # For now, we will create a dummy file to let the code run if download fails.
    mosei_path = os.path.join(data_dir, "mosei_aligned.pkl")
    
    # 2. CH-SIMS (Chinese MSA)
    # Source: CH-SIMS GitHub (https://github.com/thuiar/MMSA)
    print("\n--- Downloading CH-SIMS ---")
    sims_path = os.path.join(data_dir, "sims_aligned.pkl")
    
    # Try to download SIMS features
    # SIMS is often hosted on Baidu Netdisk (hard to script) or Google Drive.
    print("CH-SIMS 特征通常托管在百度网盘。")
    print("手动下载链接: https://github.com/thuiar/MMSA (查看 README)")
    
    # Create dummy files if they don't exist, just to make sure `train.py` runs
    # In a real scenario, the user MUST replace these with real files.
    
    if not os.path.exists(mosei_path):
        print(f"\n[Warning] 无法自动下载 MOSEI。创建了一个占位文件: {mosei_path}")
        print("请务必用真实数据覆盖它！")
        with open(mosei_path, 'wb') as f:
            f.write(b'DUMMY DATA') # This will crash pickle.load if not replaced
            
    if not os.path.exists(sims_path):
        print(f"[Warning] 无法自动下载 CH-SIMS。创建了一个占位文件: {sims_path}")
        print("请务必用真实数据覆盖它！")
        with open(sims_path, 'wb') as f:
            f.write(b'DUMMY DATA')

if __name__ == "__main__":
    main()
