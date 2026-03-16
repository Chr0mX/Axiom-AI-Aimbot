import os
import shutil
import re
import glob
import serial.tools.list_ports

def find_boards_txt():
    """Find the location of boards.txt of Arduino IDE"""
    possible_paths = []
    
    # 1. AppData (Arduino IDE 2.0+ / Board Manager)
    local_appdata = os.environ.get('LOCALAPPDATA', '')
    if local_appdata:
        # Search all versions, take the latest
        pattern = os.path.join(local_appdata, r'Arduino15\packages\arduino\hardware\avr\*\boards.txt')
        found = glob.glob(pattern)
        if found:
            # Sort and take the one with the largest version number
            possible_paths.extend(sorted(found, reverse=True))

    # 2. Program Files (Legacy IDE)
    program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')
    program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
    
    possible_paths.append(os.path.join(program_files_x86, r'Arduino\hardware\arduino\avr\boards.txt'))
    possible_paths.append(os.path.join(program_files, r'Arduino\hardware\arduino\avr\boards.txt'))

    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def spoof_arduino_board():
    """Modify boards.txt to spoof Arduino Leonardo"""
    boards_file = find_boards_txt()
    if not boards_file:
        raise FileNotFoundError("Could not find boards.txt, please ensure Arduino IDE or AVR Boards package is installed.")

    # Backup
    backup_file = boards_file + ".bak"
    if not os.path.exists(backup_file):
        shutil.copy2(boards_file, backup_file)

    try:
        with open(boards_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        spoofed = False
        
        # Spoof target: Logitech G502 HERO
        # VID: 0x046D
        # PID: 0xC07D
        
        for line in lines:
            line_strip = line.strip()
            
            # Modify VID
            if line_strip.startswith('leonardo.build.vid='):
                new_lines.append('leonardo.build.vid=0x046D\n')
                continue
                
            # Modify PID
            if line_strip.startswith('leonardo.build.pid='):
                new_lines.append('leonardo.build.pid=0xC07D\n')
                continue
                
            # Modify product name
            if line_strip.startswith('leonardo.build.usb_product='):
                new_lines.append('leonardo.build.usb_product="Logitech G502 HERO Gaming Mouse"\n')
                continue
                
            new_lines.append(line)

        # Write back to file
        with open(boards_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        return True, boards_file
        
    except Exception as e:
        # If an error occurs, try to restore
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, boards_file)
        raise e

def verify_spoof(specific_port=None):
    """
    Verify if spoofing was successful
    Returns:
        (is_spoofed, message)
    """
    target_vid = 0x046D
    target_pid = 0xC07D
    
    ports = serial.tools.list_ports.comports()
    
    # 如果指定了端口，只檢查該端口
    if specific_port:
        target_ports = [p for p in ports if p.device == specific_port]
    else:
        target_ports = ports

    # 搜尋符合的裝置
    spoofed_device = None
    original_device = None
    
    for port in target_ports:
        # 檢查是否為偽裝後的裝置
        if port.vid == target_vid and port.pid == target_pid:
            spoofed_device = port
            break
        
        # 檢查是否仍為原始 Arduino (VID=0x2341, PID=0x8036)
        if port.vid == 0x2341 and port.pid == 0x8036:
            original_device = port

    if spoofed_device:
        return True, f"成功！在 {spoofed_device.device} 檢測到偽裝裝置：\nUID: {spoofed_device.description or 'Logitech G502'}\nVID: {spoofed_device.vid:04X} PID: {spoofed_device.pid:04X}"
    
    if original_device:
        return False, f"失敗。在 {original_device.device} 檢測到原始 Arduino：\nUID: {original_device.description}\nVID: {original_device.vid:04X} PID: {original_device.pid:04X}\n\n請確認已重新上傳韌體。"
        
    return False, "未檢測到相關裝置。請確認 Arduino 已連接。"
