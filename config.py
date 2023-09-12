import argparse
import sys
import torch
from multiprocessing import cpu_count

class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.share,
            self.api,
            self.unsupported
        ) = self.arg_parse()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse() -> tuple:
        parser = argparse.ArgumentParser()
        parser.add_argument("--share", action="store_true", help="Launch with public link")
        parser.add_argument("--api", action="store_true", help="Launch with api")
        parser.add_argument("--unsupported", action="store_true", help="Enable unsupported feature")
        cmd_opts = parser.parse_args()

        return (
            cmd_opts.share,
            cmd_opts.api,
            cmd_opts.unsupported
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("INFO: Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                print("INFO: Found GPU", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif self.has_mps():
            print("INFO: No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
        else:
            print("INFO: No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
