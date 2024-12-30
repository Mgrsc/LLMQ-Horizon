from pathlib import Path
import socketserver
import http.server
import signal
import os
import sys

root_path = Path(__file__).resolve().parents[0]
temp_server_dir = root_path / "temp_server"
port = 5000
host = "0.0.0.0"  # 允许外部访问

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"访问日志: {self.address_string()} - {format%args}")

def signal_handler(signum, frame):
    """处理退出信号"""
    print("\n正在关闭服务器...")
    raise KeyboardInterrupt

class ThreadedHTTPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def start_temp_server():
    """
    临时 HTTP 文件服务器 (多线程)。
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        os.makedirs(temp_server_dir, exist_ok=True)
        os.chdir(temp_server_dir)

        handler = CustomHTTPRequestHandler
        server = ThreadedHTTPServer((host, port), handler)

        print(f"临时文件服务器已启动 (多线程)")
        print(f"监听地址: {host}")
        print(f"端口: {port}")
        print(f"目录: {temp_server_dir}")
        print("按 Ctrl+C 可以关闭服务器")

        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("服务器已安全关闭")
        sys.exit(0)
    except PermissionError:
        print("错误：没有权限启动服务器，请检查端口权限或尝试使用sudo运行")
        sys.exit(1)
    except OSError as e:
        print(f"错误：启动服务器失败 - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_temp_server()