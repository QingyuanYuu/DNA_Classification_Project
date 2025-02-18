import os
import pandas as pd
from Bio import Entrez, SeqIO

# 设置 NCBI 访问邮箱（必须填写一个有效邮箱）
Entrez.email = "jaaasn_yu@outlook.com"

# 定义数据存储路径
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "E_coli_DNA_sequences.csv")

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def download_ecoli_genome(accession="U00096.3"):
    """
    从 NCBI 下载 E. coli K-12 MG1655 的 GenBank 文件，并保存到本地
    """
    print(f"📥 Downloading E. coli genome (Accession: {accession})...")
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()

    # 保存原始数据
    gb_file = os.path.join(RAW_DATA_DIR, f"{accession}.gb")
    with open(gb_file, "w") as f:
        SeqIO.write(record, f, "genbank")
    
    print(f"✅ Genome downloaded and saved to {gb_file}")
    return record


def extract_dna_sequences(record):
    """
    提取 E. coli 基因组的编码区（CDS）和非编码区
    """
    genome_seq = record.seq
    coding_regions = []
    non_coding_regions = []

    # 获取所有 CDS 位置
    cds_positions = []
    for feature in record.features:
        if feature.type == "CDS":
            start = int(feature.location.start)
            end = int(feature.location.end)
            cds_positions.append((start, end))
            coding_regions.append(str(genome_seq[start:end]))

    # 提取非编码区（在 CDS 之间）
    cds_positions.sort()
    prev_end = 0
    for start, end in cds_positions:
        if start > prev_end:  # 计算非编码区
            non_coding_regions.append(str(genome_seq[prev_end:start]))
        prev_end = end

    print(f"✅ Extracted {len(coding_regions)} coding sequences and {len(non_coding_regions)} non-coding sequences.")
    return coding_regions, non_coding_regions


def save_to_csv(coding_regions, non_coding_regions):
    """
    将提取的 DNA 片段保存为 CSV 文件
    """
    df_coding = pd.DataFrame({"sequence": coding_regions, "label": 1})  # 1 = coding
    df_non_coding = pd.DataFrame({"sequence": non_coding_regions, "label": 0})  # 0 = non-coding
    df = pd.concat([df_coding, df_non_coding], ignore_index=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data saved to {OUTPUT_FILE}")


def main():
    """
    运行整个数据处理流程：
    1. 下载基因组数据
    2. 提取编码区 & 非编码区
    3. 保存为 CSV 文件
    """
    record = download_ecoli_genome()
    coding_regions, non_coding_regions = extract_dna_sequences(record)
    save_to_csv(coding_regions, non_coding_regions)


if __name__ == "__main__":
    main()
