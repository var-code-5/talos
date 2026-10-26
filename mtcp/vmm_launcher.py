"""Keep the VMM launcher tied to its controller even on SIGKILL/crash.

This separate executable avoids preexec_fn in the multithreaded volunteer.
"""
import ctypes
import os
import signal
import sys


def main():
    expected_parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set launcher parent-death signal")
    # Cover the race in which the controller exited before prctl took effect.
    if os.getppid() != expected_parent:
        sys.exit(1)
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
