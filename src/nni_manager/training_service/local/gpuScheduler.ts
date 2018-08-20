/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as nodeNvidiaSmi from 'node-nvidia-smi';
import { delay } from '../../common/utils';
import { GPUInfo, GPUSummary } from '../common/gpuData';

/* Example of nvidia-smi result
{
    "nvidia_smi_log": {
        "timestamp": "Fri Jul 13 15:17:27 2018",
        "driver_version": "396.26",
        "attached_gpus": "8",
        "gpu": [
            ...,
            {
                ...
                "minor_number": "5",
                "utilization": {
                    "gpu_util": "100 %",
                    "memory_util": "27 %",
                    "encoder_util": "0 %",
                    "decoder_util": "0 %"
                },
                ...
                "processes": {
                    "process_info": {
                        "pid": "39943",
                        "type": "C",
                        "process_name": "python",
                        "used_memory": "16229 MiB"
                    }
                },
                ...
            },
            {
                "$": {
                    "id": "00000000:8E:00.0"
                },
                "product_name": "Tesla P100-PCIE-16GB",
                "product_brand": "Tesla",
                "display_mode": "Enabled",
                "display_active": "Disabled",
                "persistence_mode": "Disabled",
                "accounting_mode": "Disabled",
                "accounting_mode_buffer_size": "4000",
                "driver_model": {
                    "current_dm": "N/A",
                    "pending_dm": "N/A"
                },
                "serial": "0321017108732",
                "uuid": "GPU-df3e8a0a-ce99-350c-b196-c3775eb32309",
                "minor_number": "6",
                "vbios_version": "86.00.40.00.01",
                "multigpu_board": "No",
                "board_id": "0x8e00",
                "gpu_part_number": "900-2H400-0300-031",
                "inforom_version": {
                    "img_version": "H400.0201.00.08",
                    "oem_object": "1.1",
                    "ecc_object": "4.1",
                    "pwr_object": "N/A"
                },
                "gpu_operation_mode": {
                    "current_gom": "N/A",
                    "pending_gom": "N/A"
                },
                "gpu_virtualization_mode": {
                    "virtualization_mode": "None"
                },
                "ibmnpu": {
                    "relaxed_ordering_mode": "N/A"
                },
                "pci": {
                    "pci_bus": "8E",
                    "pci_device": "00",
                    "pci_domain": "0000",
                    "pci_device_id": "15F810DE",
                    "pci_bus_id": "00000000:8E:00.0",
                    "pci_sub_system_id": "118F10DE",
                    "pci_gpu_link_info": {
                        "pcie_gen": {
                            "max_link_gen": "3",
                            "current_link_gen": "3"
                        },
                        "link_widths": {
                            "max_link_width": "16x",
                            "current_link_width": "16x"
                        }
                    },
                    "pci_bridge_chip": {
                        "bridge_chip_type": "N/A",
                        "bridge_chip_fw": "N/A"
                    },
                    "replay_counter": "0",
                    "tx_util": "0 KB/s",
                    "rx_util": "0 KB/s"
                },
                "fan_speed": "N/A",
                "performance_state": "P0",
                "clocks_throttle_reasons": {
                    "clocks_throttle_reason_gpu_idle": "Not Active",
                    "clocks_throttle_reason_applications_clocks_setting": "Not Active",
                    "clocks_throttle_reason_sw_power_cap": "Not Active",
                    "clocks_throttle_reason_hw_slowdown": "Not Active",
                    "clocks_throttle_reason_hw_thermal_slowdown": "Not Active",
                    "clocks_throttle_reason_hw_power_brake_slowdown": "Not Active",
                    "clocks_throttle_reason_sync_boost": "Not Active",
                    "clocks_throttle_reason_sw_thermal_slowdown": "Not Active"
                },
                "fb_memory_usage": {
                    "total": "16280 MiB",
                    "used": "16239 MiB",
                    "free": "41 MiB"
                },
                "bar1_memory_usage": {
                    "total": "16384 MiB",
                    "used": "2 MiB",
                    "free": "16382 MiB"
                },
                "compute_mode": "Default",
                "utilization": {
                    "gpu_util": "0 %",
                    "memory_util": "0 %",
                    "encoder_util": "0 %",
                    "decoder_util": "0 %"
                },
                "encoder_stats": {
                    "session_count": "0",
                    "average_fps": "0",
                    "average_latency": "0"
                },
                "ecc_mode": {
                    "current_ecc": "Enabled",
                    "pending_ecc": "Enabled"
                },
                "ecc_errors": {
                    "volatile": {
                        "single_bit": {
                            "device_memory": "0",
                            "register_file": "0",
                            "l1_cache": "N/A",
                            "l2_cache": "0",
                            "texture_memory": "0",
                            "texture_shm": "0",
                            "cbu": "N/A",
                            "total": "0"
                        },
                        "double_bit": {
                            "device_memory": "0",
                            "register_file": "0",
                            "l1_cache": "N/A",
                            "l2_cache": "0",
                            "texture_memory": "0",
                            "texture_shm": "0",
                            "cbu": "N/A",
                            "total": "0"
                        }
                    },
                    "aggregate": {
                        "single_bit": {
                            "device_memory": "0",
                            "register_file": "0",
                            "l1_cache": "N/A",
                            "l2_cache": "0",
                            "texture_memory": "0",
                            "texture_shm": "0",
                            "cbu": "N/A",
                            "total": "0"
                        },
                        "double_bit": {
                            "device_memory": "0",
                            "register_file": "0",
                            "l1_cache": "N/A",
                            "l2_cache": "0",
                            "texture_memory": "0",
                            "texture_shm": "0",
                            "cbu": "N/A",
                            "total": "0"
                        }
                    }
                },
                "retired_pages": {
                    "multiple_single_bit_retirement": {
                        "retired_count": "0",
                        "retired_page_addresses": "\n\t\t\t\t"
                    },
                    "double_bit_retirement": {
                        "retired_count": "0",
                        "retired_page_addresses": "\n\t\t\t\t"
                    },
                    "pending_retirement": "No"
                },
                "temperature": {
                    "gpu_temp": "33 C",
                    "gpu_temp_max_threshold": "85 C",
                    "gpu_temp_slow_threshold": "82 C",
                    "gpu_temp_max_gpu_threshold": "N/A",
                    "memory_temp": "N/A",
                    "gpu_temp_max_mem_threshold": "N/A"
                },
                "power_readings": {
                    "power_state": "P0",
                    "power_management": "Supported",
                    "power_draw": "37.29 W",
                    "power_limit": "250.00 W",
                    "default_power_limit": "250.00 W",
                    "enforced_power_limit": "250.00 W",
                    "min_power_limit": "125.00 W",
                    "max_power_limit": "250.00 W"
                },
                "clocks": {
                    "graphics_clock": "1328 MHz",
                    "sm_clock": "1328 MHz",
                    "mem_clock": "715 MHz",
                    "video_clock": "1189 MHz"
                },
                "applications_clocks": {
                    "graphics_clock": "1189 MHz",
                    "mem_clock": "715 MHz"
                },
                "default_applications_clocks": {
                    "graphics_clock": "1189 MHz",
                    "mem_clock": "715 MHz"
                },
                "max_clocks": {
                    "graphics_clock": "1328 MHz",
                    "sm_clock": "1328 MHz",
                    "mem_clock": "715 MHz",
                    "video_clock": "1328 MHz"
                },
                "max_customer_boost_clocks": {
                    "graphics_clock": "1328 MHz"
                },
                "clock_policy": {
                    "auto_boost": "N/A",
                    "auto_boost_default": "N/A"
                },
                "supported_clocks": {
                    "supported_mem_clock": {
                        "value": "715 MHz",
                        "supported_graphics_clock": [
                            "1328 MHz",
                            "1316 MHz",
                            "1303 MHz",
                            ...
                        ]
                    }
                },
                "processes": {
                    "process_info": {
                        "pid": "40788",
                        "type": "C",
                        "process_name": "python",
                        "used_memory": "16229 MiB"
                    }
                },
                "accounted_processes": "\n\t\t"
            },
            ...
        ]
    }
}*/

/**
 * GPUScheduler
 */
class GPUScheduler {

    private gpuSummary!: GPUSummary;
    private stopping: boolean;

    constructor() {
        this.stopping = false;
    }

    public async run(): Promise<void> {
        while (!this.stopping) {
            try {
                this.gpuSummary = await this.readGPUSummary();
            } catch (error) {
                console.error('Read GPU summary failed with error', error);
            }
            await delay(5000);
        }
    }

    public getAvailableGPUIndices(): number[] {
        if (this.gpuSummary !== undefined) {
            return this.gpuSummary.gpuInfos.filter((info: GPUInfo) => info.activeProcessNum === 0).map((info: GPUInfo) => info.index);
        }

        return [];
    }

    public stop(): void {
        this.stopping = true;
    }

    private readGPUSummary(): Promise<GPUSummary> {
        return new Promise((resolve: Function, reject: Function): void => {
            nodeNvidiaSmi((error: Error, data: nodeNvidiaSmi.GPUInfo) => {
                if (error !== undefined) {
                    reject(error);
                } else {
                    const gpuSummary: GPUSummary = new GPUSummary(
                        parseInt(data.nvidia_smi_log.attached_gpus, 10),
                        Date().toString(),
                        data.nvidia_smi_log.gpu.map((gpuInfo: {
                            minor_number: string;
                            utilization: {
                                gpu_util: string;
                                memory_util: string;
                            };
                            process: string | object;
                        }) => new GPUInfo(
                            typeof gpuInfo.process === 'object' ? 1 : 0,
                            parseFloat(gpuInfo.utilization.memory_util),
                            parseFloat(gpuInfo.utilization.gpu_util),
                            parseInt(gpuInfo.minor_number, 10)
                        ))
                    );
                    resolve(gpuSummary);
                }
            });
        });
    }
}

export { GPUScheduler };
