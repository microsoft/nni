declare module 'node-nvidia-smi' {   
    function smi(callback: (error: Error, data: smi.GPUInfo) => void): void;

    namespace smi {
        interface GPUInfo {
            nvidia_smi_log: {
                attached_gpus: string;
                gpu: {
                    minor_number: string;
                    utilization: {
                        gpu_util: string;
                        memory_util: string;
                    };
                    process: string | object;
                }[];
            };
        }
    }
    
    export = smi;
}