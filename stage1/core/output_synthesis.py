"""output_synthesis.py — Synthesize NFP, LTO, and generate meaningful pixel frames"""

import numpy as np

class OutputSynthesizer:
    """Rule-based output synthesis from 7 inputs + pixel generation"""
    
    @staticmethod
    def synthesize_nfp(M1, M2, M3, binary, geometry, language, V):
        """Next Frame Prediction: what happens next"""
        if M1 > 1.5:
            m1_state = "unbounded exploration"
        elif M1 > 1.0:
            m1_state = "active thinking"
        else:
            m1_state = "dormant"
        
        if M2 > 0.95:
            m2_state = "strict efficiency"
        elif M2 > 0.8:
            m2_state = "balanced efficiency"
        else:
            m2_state = "loose efficiency"
        
        if M3 > 0.9:
            m3_state = "meta-control engaged"
        elif M3 > 0.7:
            m3_state = "light meta-adjustment"
        else:
            m3_state = "minimal intervention"
        
        if V > 0.6:
            action = "will converge"
        elif V > 0.4:
            action = "will explore"
        else:
            action = "will reset"
        
        progress = int(V * 100)
        nfp = f"Entry {int(M1*2)} {action} via {m1_state} under {m2_state} with {m3_state} at {progress}% progress."
        return nfp
    
    @staticmethod
    def synthesize_lto(M1, M2, M3, binary, geometry, language, V):
        """Language Token Output: full sentence describing the semantic action"""
        
        m1_desc = "unbounded thinking" if M1 > 1.3 else "controlled reasoning"
        m2_desc = "strict optimization" if M2 > 0.95 else "adaptive optimization"
        m3_desc = "meta-oversight" if M3 > 0.8 else "minimal oversight"
        
        if language > 0.7:
            context = "with high semantic content"
        elif language > 0.3:
            context = "with moderate semantic depth"
        else:
            context = "with minimal language processing"
        
        if V > 0.65:
            action_desc = "converging toward solution"
        elif V > 0.45:
            action_desc = "exploring the solution space"
        else:
            action_desc = "diverging and resetting"
        
        lto = f"System performs {m1_desc} combined with {m2_desc} and {m3_desc} {context}, currently {action_desc}."
        return lto
    
    @staticmethod
    def generate_pixel_frame(M1, M2, M3, V, binary, geometry, language, width=64, height=64):
        """
        Generate a 64x64 pixel frame representing the 7-input state.
        
        Layout:
        - Top-left quadrant: M1 (unbounded exploration) → red
        - Top-right quadrant: M2 (efficiency) → green  
        - Bottom-left quadrant: M3 (meta-control) → blue
        - Bottom-right quadrant: V (value) → brightness
        - Center: 7-input interaction patterns
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        h2, w2 = height // 2, width // 2
        
        # Top-left: M1 intensity (unbounded exploration)
        frame[:h2, :w2, 0] = int(min(max(M1 * 100, 0), 255))
        
        # Top-right: M2 intensity (efficiency)
        frame[:h2, w2:, 1] = int(min(max(M2 * 255, 0), 255))
        
        # Bottom-left: M3 intensity (meta-control)
        frame[h2:, :w2, 2] = int(min(max(M3 * 255, 0), 255))
        
        # Bottom-right: V brightness (value function) - mix all channels
        frame[h2:, w2:] = int(min(max(V * 255, 0), 255))
        
        # Center circle: 7-input interaction
        cy, cx = height // 2, width // 2
        radius = int(8 + V * 12)  # Grows with V
        y, x = np.ogrid[:height, :width]
        center_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Center color is weighted average of all 7 inputs
        center_color = np.array([
            min(max(int(M1 * 50 + binary * 100), 0), 255),  # Red channel
            min(max(int(M2 * 150 + geometry * 100), 0), 255),  # Green channel
            min(max(int(M3 * 150 + language * 100), 0), 255)   # Blue channel
        ], dtype=np.uint8)
        frame[center_mask] = center_color
        
        # Add rings based on input activity
        if binary > 0.5:
            ring1 = (x - cx)**2 + (y - cy)**2 == (radius + 2)**2
            frame[ring1] = [255, 0, 0]  # Red ring
        
        if geometry > 0.5:
            ring2 = (x - cx)**2 + (y - cy)**2 == (radius + 4)**2
            frame[ring2] = [0, 255, 0]  # Green ring
        
        if language > 0.5:
            ring3 = (x - cx)**2 + (y - cy)**2 == (radius + 6)**2
            frame[ring3] = [0, 0, 255]  # Blue ring
        
        return frame
    
    @staticmethod
    def synthesize(M1, M2, M3, binary, geometry, language, V):
        """Synthesize outputs and generate pixel frame"""
        nfp = OutputSynthesizer.synthesize_nfp(M1, M2, M3, binary, geometry, language, V)
        lto = OutputSynthesizer.synthesize_lto(M1, M2, M3, binary, geometry, language, V)
        frame = OutputSynthesizer.generate_pixel_frame(M1, M2, M3, V, binary, geometry, language)
        
        return nfp, lto, frame
