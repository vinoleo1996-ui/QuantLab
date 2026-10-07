"""免费数据驱动的 Quant Data OS。

原始字节只追加。清洗、复权、映射和时点过滤都写新版本。
无法核对公布时间的记录留在银层并标成 research_grade=false，不进入训练集。
"""

from quantlab.data_os.pipeline import build_dataset

__all__ = ["build_dataset"]
