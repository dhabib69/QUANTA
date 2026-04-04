import sys
import ctypes
import os

def isolate_proxy():
    print("=" * 70)
    print("🛡️ QUANTA - Proxy Isolator")
    print("=" * 70)
    print("Isolating Psiphon Proxy to ONLY tunnel the QUANTA Bot...")
    
    if sys.platform != 'win32':
        print("❌ This script is only meant for Windows.")
        return

    try:
        import winreg
        # Access Windows Registry for Internet Settings
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE)
        
        # Turn off the system-wide proxy switch
        winreg.SetValueEx(registry_key, "ProxyEnable", 0, winreg.REG_DWORD, 0)
        winreg.CloseKey(registry_key)
        
        # Force Windows to apply the changes immediately without restarting
        internet_option_settings_changed = 39
        internet_option_refresh = 37
        internet_set_option = ctypes.windll.wininet.InternetSetOptionW
        internet_set_option(0, internet_option_settings_changed, 0, 0)
        internet_set_option(0, internet_option_refresh, 0, 0)
        
        print("\n✅ SUCCESS: System Proxy Disabled!")
        print("⚡ Brave Browser and other apps will now use your fast Direct Connection.")
        print("🤖 QUANTA Bot will securely continue using the proxy internally.\n")
    except Exception as e:
        print(f"\n❌ Failed to isolate proxy: {e}\n")

if __name__ == "__main__":
    isolate_proxy()
    os.system("pause")
