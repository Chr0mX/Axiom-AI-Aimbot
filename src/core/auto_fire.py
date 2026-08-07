# auto_fire.py
"""Auto-fire functionality module - handles automatic shooting logic"""

from __future__ import annotations

import queue
import time
import traceback
import logging
from typing import TYPE_CHECKING

from win_utils import is_key_pressed, send_mouse_click

if TYPE_CHECKING:
    from .config import Config


def auto_fire_loop(config: Config, boxes_queue: queue.Queue) -> None:
    """Independent loop for auto-fire functionality
    
    Listens for the auto-fire key and automatically triggers shooting when the crosshair is 
    within the detected target range. Supports three shooting modes: head, body, and all areas.
    
    Args:
        config: Config instance containing auto-fire related settings
        boxes_queue: Bounding box queue, retrieves target positions from the AI inference loop
    
    Note:
        This function should run in an independent daemon thread
    """
    last_key_state = False
    delay_start_time = None
    last_fire_time = 0
    cached_boxes = []
    last_box_update = 0
    logger = logging.getLogger(__name__)
    
    BOX_UPDATE_INTERVAL = 1 / 60  # 60Hz update frequency
    
    # Cache key configuration
    auto_fire_key = config.auto_fire_key
    auto_fire_key2 = getattr(config, 'auto_fire_key2', None)
    last_key_update = 0
    key_update_interval = 0.5  # Check for key configuration changes every 0.5 seconds
    
    while config.Running:
        try:
            current_time = time.time()
            
            # Periodically update key configuration
            if current_time - last_key_update > key_update_interval:
                auto_fire_key = config.auto_fire_key
                auto_fire_key2 = getattr(config, 'auto_fire_key2', None)
                last_key_update = current_time
            
            # Check key state
            key_state = bool(getattr(config, 'always_auto_fire', False)) or is_key_pressed(auto_fire_key)
            if auto_fire_key2:
                key_state = key_state or is_key_pressed(auto_fire_key2)

            # Handle key state changes
            if key_state and not last_key_state:
                delay_start_time = current_time
            
            if key_state:
                # Check ads delay
                if delay_start_time and (current_time - delay_start_time >= config.auto_fire_delay):
                    # Check shooting cooldown
                    if current_time - last_fire_time >= config.auto_fire_interval:
                        
                        # Update bounding box cache
                        if current_time - last_box_update >= BOX_UPDATE_INTERVAL:
                            try:
                                # Get latest detection results from queue (using get_nowait instead of direct access)
                                if not boxes_queue.empty():
                                    cached_boxes = boxes_queue.get_nowait()
                                    last_box_update = current_time
                            except queue.Empty:
                                # No new data, use old cache
                                pass
                            except Exception as e:
                                logger.warning("AutoFire failed to read detection queue: %s", e)
                        
                        # Determine if shooting should occur
                        if cached_boxes:
                            crosshair_x, crosshair_y = config.crosshairX, config.crosshairY
                            target_part = config.auto_fire_target_part
                            head_height_ratio = config.head_height_ratio
                            head_width_ratio = config.head_width_ratio
                            body_width_ratio = config.body_width_ratio
                            
                            # Shooting judgment
                            should_fire = False
                            for box in cached_boxes:
                                x1, y1, x2, y2 = box
                                box_w, box_h = x2 - x1, y2 - y1
                                box_center_x = x1 + box_w * 0.5
                                
                                # Bound checks
                                if target_part == "head":
                                    head_h = box_h * head_height_ratio
                                    head_w = box_w * head_width_ratio
                                    head_x1 = box_center_x - head_w * 0.5
                                    head_x2 = box_center_x + head_w * 0.5
                                    head_y2 = y1 + head_h
                                    should_fire = (head_x1 <= crosshair_x <= head_x2 and y1 <= crosshair_y <= head_y2)
                                elif target_part == "body":
                                    body_w = box_w * body_width_ratio
                                    body_x1 = box_center_x - body_w * 0.5
                                    body_x2 = box_center_x + body_w * 0.5
                                    body_y1 = y1 + box_h * head_height_ratio
                                    should_fire = (body_x1 <= crosshair_x <= body_x2 and body_y1 <= crosshair_y <= y2)
                                elif target_part == "both":
                                    # 檢查頭部和身體區域
                                    head_h = box_h * head_height_ratio
                                    head_w = box_w * head_width_ratio
                                    head_x1 = box_center_x - head_w * 0.5
                                    head_x2 = box_center_x + head_w * 0.5
                                    
                                    is_in_head = (head_x1 <= crosshair_x <= head_x2 and y1 <= crosshair_y <= y1 + head_h)
                                    
                                    if not is_in_head:
                                        body_w = box_w * body_width_ratio
                                        body_x1 = box_center_x - body_w * 0.5
                                        body_x2 = box_center_x + body_w * 0.5
                                        body_y1 = y1 + head_h
                                        is_in_body = (body_x1 <= crosshair_x <= body_x2 and body_y1 <= crosshair_y <= y2)
                                        should_fire = is_in_body
                                    else:
                                        should_fire = True

                                if should_fire:
                                    # 執行射擊
                                    mouse_click_method = getattr(config, 'mouse_click_method', 'mouse_event')
                                    send_mouse_click(mouse_click_method)
                                    last_fire_time = current_time
                                    break
            else:
                delay_start_time = None
                if cached_boxes:
                    cached_boxes = []

            last_key_state = key_state
            
            time.sleep(1 / 60)
            
        except Exception as e:
            logger.error("AutoFire 發生錯誤: %s", e)
            traceback.print_exc()
            time.sleep(1.0)

