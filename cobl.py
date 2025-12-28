import numpy as np

class CObLCompositor:
    def __init__(self, delta=1e-6):
        self.delta = delta

    def recursive_compositing(self, layers_x, layers_m):
        """
        Triển khai Eq. (1) trong bài báo CObL:
        Hợp nhất đệ quy các lớp vật thể và nền.
        """
        if not layers_x:
            return None, []

        h, w, c = layers_x[0].shape
        
        # Khởi tạo accumulator (x_bar, m_bar)
        x_bar = np.zeros((h, w, c), dtype=np.float32)
        m_bar = np.zeros((h, w), dtype=np.float32)
        
        process_steps = []
        
        for i in range(len(layers_x)):
            x_i = layers_x[i] # Màu lớp i
            m_i = layers_m[i] # Alpha lớp i
            
            # Xử lý chiều dữ liệu mask
            if m_i.ndim == 2: 
                m_i_3d = np.stack([m_i]*3, axis=-1)
            else: 
                m_i_3d = m_i
                
            # --- CÔNG THỨC CHÍNH (Eq. 1) ---
            # 1. Cập nhật Alpha tích lũy
            m_bar_new = m_bar * (1 - m_i) + m_i
            
            # 2. Cập nhật Màu tích lũy
            term_new = x_i * m_i_3d
            term_prev = x_bar * np.stack([m_bar]*3, axis=-1) * (1 - m_i_3d)
            
            # Tránh chia cho 0 bằng delta
            denominator = np.stack([m_bar_new]*3, axis=-1) + self.delta
            x_bar_new = (term_new + term_prev) / denominator
            
            # Cập nhật trạng thái
            x_bar = x_bar_new
            m_bar = m_bar_new
            
            process_steps.append(np.clip(x_bar, 0, 1))
            
        return x_bar, process_steps