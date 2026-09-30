import resource
import time

class ResourceTracker:
    def __enter__(self):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.start_wall = time.perf_counter()
        self.start_user = usage.ru_utime
        self.start_kernel = usage.ru_stime
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)

        end_wall = time.perf_counter()
        end_user = usage.ru_utime
        end_kernel = usage.ru_stime

        total_wall = end_wall - self.start_wall
        total_user = end_user - self.start_user
        total_kernel = end_kernel - self.start_kernel

        memory = usage.ru_maxrss / 1024

        print(f"User Time {total_user:.2f} s, Kernel Time {total_kernel:.2f} s, Wall Time {total_wall:.2f} s, Peak Memory {memory:.2f} MiB")
