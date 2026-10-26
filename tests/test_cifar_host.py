import types
import unittest
from unittest.mock import patch
from mtcp import host


class HostTests(unittest.TestCase):
    def test_macos_fallback_reports_no_hard_affinity(self):
        fake_os=types.SimpleNamespace(cpu_count=lambda:4)
        fake_psutil=types.SimpleNamespace(Process=lambda:types.SimpleNamespace())
        with patch.object(host,'os',fake_os),patch.object(host,'_psutil',return_value=fake_psutil):
            self.assertEqual(host.available_cpus(),[0,1,2,3])
            self.assertFalse(host.pin_cpus([0]))

    def test_windows_affinity_and_peak_working_set(self):
        assigned=[]
        def affinity(cpus=None):
            if cpus is not None:assigned.extend(cpus)
            return [1,3]
        process=types.SimpleNamespace(cpu_affinity=affinity,
            memory_info=lambda:types.SimpleNamespace(peak_wset=256*1024**2,rss=128*1024**2))
        fake_psutil=types.SimpleNamespace(Process=lambda:process)
        with patch.object(host,'os',types.SimpleNamespace(cpu_count=lambda:8)),patch.object(host,'_psutil',return_value=fake_psutil):
            self.assertEqual(host.available_cpus(),[1,3]);self.assertTrue(host.pin_cpus([3]));self.assertEqual(assigned,[3])
            with patch.dict('sys.modules',{'resource':None}):self.assertEqual(host.peak_rss_mb(),256)

    def test_portable_memory_units(self):
        fake=types.SimpleNamespace(virtual_memory=lambda:types.SimpleNamespace(total=8*1024**3,available=3*1024**3))
        with patch.object(host,'_psutil',return_value=fake):self.assertEqual(host.memory_mb(),(8192,3072))
