# session_utils.py
"""ONNX 運行時會話優化模組 - 提供推理性能優化選項"""

import logging
import onnxruntime as ort


def optimize_onnx_session(config):
    """優化 ONNX 運行時設定
    
    創建並配置 ONNX 會話選項，啟用圖優化和記憶體優化。
    
    Args:
        config: 配置實例（目前未使用，保留供未來擴展）
        
    Returns:
        ort.SessionOptions: 優化後的會話選項實例
        None: 當優化設定失敗時
    """
    logger = logging.getLogger(__name__)
    try:
        # 設定ONNX運行時選項
        session_options = ort.SessionOptions()
        
        # 啟用所有優化
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 設置執行模式為順序（對單個請求更快）
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 啟用記憶體優化
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        # 盡量降低 CPU 背景占用（DirectML 主要在 GPU 上跑）
        # 注意：部分版本/提供者可能忽略此設定，但通常是安全的。
        try:
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
        except Exception as e:
            logger.warning("ONNX 執行緒參數設定失敗: %s", e)

        # 避免 thread spinning 造成高 CPU（若版本支援）
        try:
            session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception as e:
            logger.warning("ONNX allow_spinning 設定失敗: %s", e)
        
        return session_options
        
    except Exception as e:
        logger.error("ONNX 優化失敗: %s", e)
        return None

