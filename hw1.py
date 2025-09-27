def generate_pam(x, input_path, output_path):
    import pandas as pd
    import numpy as np
    # 1) 讀 mut.txt；跳過以 # 開頭的註解；第一欄為 row index
    df = pd.read_csv(input_path, delim_whitespace=True, index_col=0, comment="#")

    # 2) 轉為機率矩陣：除以 10000
    M1 = df.to_numpy(dtype=float) / 10000.0

    # 3) PAM^x：矩陣乘方（不是元素次方）
    Mx = np.linalg.matrix_power(M1, int(x))

    # 4) 背景頻率 f_i（依投影片，可用 Dayhoff 常用值；順序依 row index）
    dayhoff_fi = {
        "A": 0.087, "R": 0.041, "N": 0.040, "D": 0.047, "C": 0.033,
        "Q": 0.038, "E": 0.050, "G": 0.089, "H": 0.034, "I": 0.037,
        "L": 0.085, "K": 0.081, "M": 0.015, "F": 0.040, "P": 0.051,
        "S": 0.070, "T": 0.058, "W": 0.010, "Y": 0.030, "V": 0.065,
    }
    row_order = list(df.index)
    col_order = list(df.columns)
    # 若檔案列序與欄序不一致，這裡假設輸出仍用檔案的原欄序
    fi = np.array([dayhoff_fi[a] for a in row_order], dtype=float)

    # 5) 依投影片公式：s_ij = 10 * log10( M_ij / f_i )
    with np.errstate(divide="ignore"):
        ratio = np.clip(Mx / (fi[:, None] + 1e-300), 1e-300, None)
        s = 10.0 * np.log10(ratio)

    # 6) 四捨五入
    S = np.round(s).astype(int)

    # 7) 用 pandas.to_csv 輸出；空白分隔、保留列名與欄名
    out = pd.DataFrame(S, index=row_order, columns=col_order)
    # 第一個空白欄名：與範例格式一致
    out.to_csv(output_path, sep=" ", header=True, index=True, lineterminator="\n")



