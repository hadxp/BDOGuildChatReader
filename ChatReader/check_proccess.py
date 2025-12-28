import sys
import time
import ctypes
from ctypes import wintypes

import psutil

user32 = ctypes.windll.user32

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowThreadProcessId = user32.GetWindowThreadProcessId

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_RESTORE = 9
SW_MINIMIZE = 6

def is_window_minimized(hwnd: int) -> bool:
    return IsWindowVisible(hwnd)

def get_pid_by_name(process_name) -> int | None:
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() == process_name.lower():
            return proc.info['pid']
    return None

def get_hwnds_for_pid(pid):
    hwnds = []

    # Define the callback inside so it can access 'hwnds' list directly
    def callback(hwnd, lParam):
        # Get the PID of the window
        lpdw_pid = ctypes.c_ulong()
        GetWindowThreadProcessId(hwnd, ctypes.byref(lpdw_pid))

        if lpdw_pid.value == pid:
            hwnds.append(hwnd)
        return True

    # Call the API
    EnumWindows(EnumWindowsProc(callback), 0)
    return hwnds


def manage_window(exe_name, action="hide"):
    target_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == exe_name.lower():
            target_pid = proc.info['pid']
            break

    if not target_pid:
        print(f"Process '{exe_name}' not found.")
        return

    hwnds = get_hwnds_for_pid(target_pid)

    if not hwnds:
        print(f"No windows found for {exe_name}.")
        return

    cmd = SW_HIDE if action == "hide" else SW_RESTORE

    for hwnd in hwnds:
        ctypes.windll.user32.ShowWindow(hwnd, cmd)
        if action == "show":
            ctypes.windll.user32.SetForegroundWindow(hwnd)

    print(f"Action '{action}' applied to {len(hwnds)} window(s) of {exe_name}.")

exe_name = "notepad.exe"

#pid = get_pid_by_name(exe_name)
#print(f"Found PID: {pid}")

print("Hide")
manage_window(exe_name, action="hide")
for hwnd in get_hwnds_for_pid(get_pid_by_name(exe_name)):
    print(f"Hidden: {is_window_minimized(hwnd)}")

time.sleep(2)
print("Restore")
manage_window(exe_name, action="show")
for hwnd in get_hwnds_for_pid(get_pid_by_name(exe_name)):
    print(f"Hidden: {is_window_minimized(hwnd)}")

sys.exit(0)

was_window_minimized: bool = False
is_visible: bool = is_window_minimized(hwnd)
print(f"Visible: {is_visible}")

while True:
    if hwnd and is_window_minimized(hwnd):  # if the window is minimized it need to be restored, to take a screenshot
        restore_window(hwnd)
        was_window_minimized = True

    print(f"restored {was_window_minimized}")
    time.sleep(2)

    # minimize the window again, if it was minimized
    if hwnd and was_window_minimized:
        minimize_window(hwnd)

    print(f"minimized {was_window_minimized}")
    time.sleep(2)
