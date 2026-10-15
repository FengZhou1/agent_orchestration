"""Optional benchmark telemetry, not a simulator/model dependency."""
import threading
import psutil


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_private = self.peak_children_private = 0
        self.min_available = psutil.virtual_memory().available

    def __enter__(self):
        def sample():
            process = psutil.Process()
            while not self.stop.is_set():
                children = 0
                for child in process.children(recursive=True):
                    try:
                        mem = child.memory_info()
                        children += getattr(mem, 'private', mem.rss)
                    except psutil.NoSuchProcess:
                        pass
                mem = process.memory_info()
                self.peak_children_private = max(self.peak_children_private, children)
                self.peak_private = max(self.peak_private, children + getattr(mem, 'private', mem.rss))
                self.min_available = min(self.min_available, psutil.virtual_memory().available)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def report(self):
        return {'peak_private_bytes': self.peak_private,
                'peak_children_private_bytes': self.peak_children_private,
                'minimum_system_available_bytes': self.min_available}
