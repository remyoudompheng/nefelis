import importlib.resources
import json
import os
import subprocess
import logging

import vulkan as vk

logger = logging.getLogger("vulkan")

SHADER_CACHE = {}


def shader(name, defines=None, entry="main"):
    if defines is None:
        defines = {}
    defines = [f"-D{k}={v}" for k, v in defines.items()]
    key = (name,) + tuple(defines)
    if key in SHADER_CACHE:
        return SHADER_CACHE[key]
    with importlib.resources.path("nefelis.vulkan", name + ".comp") as srcpath:
        p = subprocess.Popen(
            ["glslc", "--target-env=vulkan1.3", "-I."]
            + defines
            + [f"-fentry-point={entry}", "-fshader-stage=compute", str(srcpath), "-o-"],
            cwd=os.path.dirname(srcpath),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        logger.debug(f"Exec {' '.join(p.args)}")
        out, err = p.communicate(timeout=1)
    if p.returncode:
        raise EnvironmentError(f"shader compilation failed: code {p.returncode}")
    SHADER_CACHE[key] = out
    return out


class GPUInfo:
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.loaded = True
        try:
            self.vulkaninfo()
            logger.debug("Loaded information from libvulkan")
        except Exception:
            pass

    def vulkaninfo(self):
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            apiVersion=vk.VK_API_VERSION_1_0,
        )

        createInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=appInfo,
            enabledExtensionCount=1,
            ppEnabledExtensionNames=[
                vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
            ],
        )

        instance = vk.vkCreateInstance(createInfo, None)

        # FIXME: read other devices?
        dev = vk.vkEnumeratePhysicalDevices(instance)[0]
        deviceExts = [
            e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(dev, None)
        ]

        ffi = vk.ffi
        props = vk.VkPhysicalDeviceProperties2()
        vendor = None
        if "VK_AMD_shader_core_properties2" in deviceExts:
            vendor = ffi.new("VkPhysicalDeviceShaderCoreProperties2AMD *")
            vendor.sType = (
                vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD
            )
        elif "VK_NV_shader_sm_builtins" in deviceExts:
            vendor = ffi.new("VkPhysicalDeviceShaderSMBuiltinsPropertiesNV *")
            vendor.sType = (
                vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV
            )
        props.pNext = vendor

        vk.vkGetPhysicalDeviceProperties2(dev, props)
        self.devname = ffi.string(props.properties.deviceName).decode()
        self.devtype = props.properties.deviceType
        self.stamp_period = props.properties.limits.timestampPeriod
        self.max_shmem = props.properties.limits.maxComputeSharedMemorySize

        if "VK_AMD_shader_core_properties2" in deviceExts:
            # For AMD the number of cores is the number of CU (2 CU per RDNA WGP)
            self.gpu_cores = vendor.activeComputeUnitCount
        elif "VK_NV_shader_sm_builtins" in deviceExts:
            # For NVIDIA the number of cores is the number of SM
            self.gpu_cores = vendor.shaderSMCount
        else:
            # For other vendors, use a reasonable arbitrary value.
            # ARM has VK_ARM_shader_core_builtins
            self.gpu_cores = (
                32 if self.devtype == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 8
            )
            logger.warning("Unknown number of GPU cores, assuming {self.gpu_cores}")
        devtype_str = {
            vk.VK_PHYSICAL_DEVICE_TYPE_OTHER: "other",
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "iGPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "dGPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "vGPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: "CPU",
        }
        logger.info(
            f"Vulkan device {self.devname} ({devtype_str[self.devtype]}) with {self.gpu_cores} cores"
        )
        logger.debug(
            f"Properties stamp_period={self.stamp_period} shmem={self.max_shmem}"
        )


_gpuinfo = GPUInfo()


def device_name() -> str:
    _gpuinfo.load()
    return _gpuinfo.devname


def is_discrete_gpu() -> bool:
    _gpuinfo.load()
    return _gpuinfo.devtype == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU


def stamp_period() -> int:
    _gpuinfo.load()
    return _gpuinfo.stamp_period


def gpu_cores() -> int:
    _gpuinfo.load()
    return _gpuinfo.gpu_cores
