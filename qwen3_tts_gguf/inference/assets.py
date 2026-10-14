"""
assets.py - 资产管理器
负责加载词表、投影矩阵和 Codec 嵌入表。
提供针对工匠模型加速的预投影嵌入表。
"""
import os
import numpy as np
from . import logger

class AssetsManager:
    """
    负责加载和持有所有静态权重资产。
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # 定义关键资产路径
        self.paths = {
            "text_table": os.path.join(model_dir, 'embeddings', "text_embedding_projected.npy"),
            # int8 压缩版(全词表, per-row scale 反量化): 优先加载
            "text_table_int8": os.path.join(model_dir, 'embeddings', "text_embedding_projected_int8.npy"),
            "text_table_scale": os.path.join(model_dir, 'embeddings', "text_embedding_projected_int8_scale.npy"),
            "proj_w": os.path.join(model_dir, 'embeddings', "proj_weight.npy"),
            "proj_b": os.path.join(model_dir, 'embeddings', "proj_bias.npy")
        }
        
        self.load_all()

    def load_all(self):
        """加载所有权重资产到内存"""
        logger.info(f"[Assets] 正在从 {self.model_dir} 加载资产...")
        
        # 1. 加载文本投影表（优先 int8 压缩版 text_embedding_projected_int8.npy）
        int8_p, scale_p = self.paths["text_table_int8"], self.paths["text_table_scale"]
        if (not os.environ.get("Q3TTS_DISABLE_INT8_EMB")
                and os.path.exists(int8_p) and os.path.exists(scale_p)):
            q = np.load(int8_p)
            scale = np.load(scale_p).reshape(-1, 1)
            self.text_table = (q.astype(np.float32) * scale).astype(np.float16)
            logger.info(f"[Assets] text_embedding int8 反量化加载: {self.text_table.shape}")
        else:
            if not os.path.exists(self.paths["text_table"]):
                raise FileNotFoundError(f"缺失核心资产: {self.paths['text_table']}")
            self.text_table = np.load(self.paths["text_table"], 'r')
        self.tts_pad = self.text_table[151671] # 预存 PAD 向量
        
        # 2. 加载投影矩阵 (2048 -> 1024) - 可选，针对 1.7B
        has_proj = os.path.exists(self.paths["proj_w"]) and os.path.exists(self.paths["proj_b"])
        if has_proj:
            self.proj = {
                "weight": np.load(self.paths["proj_w"], 'r'),
                "bias": np.load(self.paths["proj_b"], 'r')
            }
            pw = self.proj["weight"]
            pb = self.proj["bias"]
            logger.info("[Assets] 已加载 2048->1024 投影矩阵 (针对 1.7B)")
        else:
            self.proj = None
            logger.info("[Assets] 未检测到投影矩阵，将跳过预投影 (针对 0.6B)")
        
        # 3. 加载 16 组 Codec Embedding Tables
        self.emb_tables = []
        self.emb_tables_1024 = []
        
        for i in range(16):
            # int8 压缩版优先(codec_embedding_{i}_int8.npy + per-row scale),
            # Q3TTS_DISABLE_INT8_CODEC=1 强制回退 fp32 原版(A/B 用)
            int8_p = os.path.join(self.model_dir, 'embeddings', f"codec_embedding_{i}_int8.npy")
            scale_p = os.path.join(self.model_dir, 'embeddings', f"codec_embedding_{i}_int8_scale.npy")
            if (not os.environ.get("Q3TTS_DISABLE_INT8_CODEC")
                    and os.path.exists(int8_p) and os.path.exists(scale_p)):
                q = np.load(int8_p)
                scale = np.load(scale_p).reshape(-1, 1)
                table = (q.astype(np.float32) * scale).astype(np.float32)
                logger.info(f"[Assets] codec_embedding_{i} int8 反量化加载: {table.shape}")
            else:
                path = os.path.join(self.model_dir, 'embeddings', f"codec_embedding_{i}.npy")
                if not os.path.exists(path):
                    # 如果没有 16 组，尝试兼容某些旧版本或精简版
                    logger.warning(f"[Assets] 缺失第 {i} 组 Codec 嵌入表，尝试跳过...")
                    continue
                # 文件为 fp16 存储(2026-10-11 起, 原 fp32 壳即 fp16 精度, fp16 无损省一半),
                # 运行时统一上采 fp32, 查表/矩阵乘下游不变
                table = np.load(path, 'r').astype(np.float32)

            self.emb_tables.append(table)
            
            # 预投影：针对 Predictor 模型（工匠模型）加速
            if self.proj is not None:
                # 1.7B 模式：从 2048 投影到 1024
                table_1024 = table @ pw.T + pb
                self.emb_tables_1024.append(table_1024)
            else:
                # 0.6B 模式：直接使用（已经是 1024 维度）
                self.emb_tables_1024.append(table)
            
        logger.info(f"✅ [Assets] 资产加载完成 (Codec 表数量: {len(self.emb_tables)})")

    def get_text_embedding(self, token_id: int) -> np.ndarray:
        return self.text_table[token_id]

    def get_codec_embedding(self, q_idx: int, code: int) -> np.ndarray:
        """获取原始 2048 维嵌入"""
        return self.emb_tables[q_idx][code]

    def get_codec_embedding_1024(self, q_idx: int, code: int) -> np.ndarray:
        """获取预投影后的 1024 维嵌入"""
        return self.emb_tables_1024[q_idx][code]
