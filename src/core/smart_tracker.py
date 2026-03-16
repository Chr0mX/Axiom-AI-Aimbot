import numpy as np
from typing import Tuple, Optional

class SmartTracker:
    """
    Smart Tracker
    Features:
    1. Jitter Smoothing
    2. Zero-Lag Reset for sudden stops/direction changes
    3. Discards physical inertia assumptions
    """
    
    def __init__(self, smoothing_factor: float = 0.5, stop_threshold: float = 20.0, position_deadzone: float = 4.0):
        """
        Args:
            smoothing_factor (0.0~1.0): 
                Higher values are smoother but have more lag, lower values react faster but are jitterier.
                Recommended range: 0.3 ~ 0.6.
            stop_threshold: 
                Forces velocity to zero if below this value (pixels/sec) to prevent micro-drifts.
            position_deadzone:
                Position deadzone (pixels). Stops correction when the distance between crosshair and target is 
                smaller than this value, avoiding oscillations near the target. Recommended: 3~5 pixels.
        """
        self.alpha = smoothing_factor
        self.stop_threshold = stop_threshold
        self.position_deadzone = position_deadzone
        
        # State
        self.last_x = None
        self.last_y = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False
        
    def update(self, measured_x: float, measured_y: float, dt: float) -> Tuple[float, float, float, float]:
        """Update position and calculate velocity"""
        if not self.initialized or dt <= 0:
            self.last_x = measured_x
            self.last_y = measured_y
            self.vx = 0.0
            self.vy = 0.0
            self.initialized = True
            return measured_x, measured_y, 0.0, 0.0

        # 1. Calculate raw instantaneous velocity (Raw Velocity)
        raw_vx = (measured_x - self.last_x) / dt
        raw_vy = (measured_y - self.last_y) / dt
        
        # 2. Smart filtering logic
        # Check direction change: if new velocity and old velocity are in opposite directions (dot product < 0), 
        # it means the target is ADAD or suddenly stopping.
        # In this case, we don't smooth and strictly adopt the new velocity (sacrificing smoothness for reaction speed).
        dot_product = raw_vx * self.vx + raw_vy * self.vy
        
        if dot_product < 0:
            # Direction change/sudden stop detected: reset velocity, no smoothing
            self.vx = raw_vx
            self.vy = raw_vy
        else:
            # Moving in the same direction: use Exponential Moving Average (EMA) to eliminate YOLO jitter
            self.vx = self.vx * self.alpha + raw_vx * (1 - self.alpha)
            self.vy = self.vy * self.alpha + raw_vy * (1 - self.alpha)
            
        # 3. Forced standstill (Deadzone)
        # If velocity is very small, force to zero to solve the problem of crosshair jittering on stationary targets
        if abs(self.vx) < self.stop_threshold: self.vx = 0
        if abs(self.vy) < self.stop_threshold: self.vy = 0

        # Update position records
        self.last_x = measured_x
        self.last_y = measured_y
        
        return measured_x, measured_y, self.vx, self.vy

    def is_in_deadzone(self, target_x: float, target_y: float, crosshair_x: float, crosshair_y: float) -> bool:
        """
        Check if the crosshair is already within the target's deadzone
        
        Args:
            target_x, target_y: Target position
            crosshair_x, crosshair_y: Crosshair position (usually screen center)
            
        Returns:
            True if within deadzone, no correction needed
        """
        if self.position_deadzone <= 0:
            return False
            
        dx = target_x - crosshair_x
        dy = target_y - crosshair_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        return distance < self.position_deadzone
    
    def get_corrected_move(self, target_x: float, target_y: float, 
                           crosshair_x: float, crosshair_y: float) -> Tuple[float, float]:
        """
        Calculate corrected movement, considering position deadzone
        
        Args:
            target_x, target_y: 目標位置
            crosshair_x, crosshair_y: 準心位置
            
        Returns:
            (move_x, move_y): 修正後的移動量。如果在死區內則返回 (0, 0)
        """
        dx = target_x - crosshair_x
        dy = target_y - crosshair_y
        
        # 位置死區檢查：距離太近就不動
        distance = np.sqrt(dx * dx + dy * dy)
        if distance < self.position_deadzone:
            return 0.0, 0.0
            
        return dx, dy

    def get_predicted_position(self, prediction_time: float) -> Tuple[float, float]:
        """預測未來位置"""
        if not self.initialized:
            return 0.0, 0.0
            
        # 簡單線性預測：位置 + 速度 * 時間
        pred_x = self.last_x + self.vx * prediction_time
        pred_y = self.last_y + self.vy * prediction_time
        
        return pred_x, pred_y
        
    def reset(self):
        self.last_x = None
        self.last_y = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False
