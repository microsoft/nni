// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'node:fs/promises';
import os from 'node:os';
import util from 'node:util';

import { globals } from 'common/globals';
import { getLogger } from 'common/log';
import { runPythonModule } from 'common/pythonScript';
import { getIPV4Address } from 'common/utils';

export async function collectPlatformInfo(includeGpu: boolean): Promise<Record<string, any>> {
    // TODO: explicitly set debug flag
    const detailed = (globals.args.logLevel === 'debug' || globals.args.logLevel === 'trace');

    const info: any = {};
    const errors: any = {};

    /* nni & python */
    try {
        const versionJson = await runPythonModule('nni.tools.nni_manager_scripts.collect_version_info');
        info.version = JSON.parse(versionJson);
    } catch (error) {
        errors.version = error;
    }

    /* os */
    info.system = {
        platform: process.platform,
        version: os.release(),
    };

    /* cpu */
    try {
        info.cpu = getCpuInfo(detailed);
    } catch (error) {
        errors.cpu = error;
    }

    /* gpu */
    if (includeGpu) {
        try {
            const args = detailed ? [ '--detail' ] : undefined;
            const gpuJson = await runPythonModule('nni.tools.nni_manager_scripts.collect_gpu_info', args);
            info.gpu = JSON.parse(gpuJson);
        } catch (error) {
            errors.gpu = error;
        }
    }

    /* memory */
    try {
        info.memory = {
            memory: formatSize(os.totalmem()),
            freeMemory: formatSize(os.freemem()),
            utilization: formatUtil(1 - os.freemem() / os.totalmem()),
        };
    } catch (error) {
        errors.memory = error;
    }

    /* disk */
    try {
        info.disk = await getDiskInfo();
    } catch (error) {
        errors.disk = error;
    }

    /* network */
    if (detailed) {
        try {
            const ipv4 = await getIPV4Address();
            info.network = { ipv4 };
        } catch (error) {
            errors.network = error;
        }
    }

    /* error handling */

    for (const key in errors) {
        getLogger('collectEnvironmentInfo').error(`Failed to collect ${key} info:`, errors[key]);
        info[key] = { error: util.inspect(errors[key]) };
    }

    return info;
}

function getCpuInfo(detailed: boolean): any {
    const ret: any = {};
    const cpus = os.cpus();
    if (detailed) {
        const models = cpus.map(cpu => cpu.model);
        const dedup = Array.from(new Set(models));
        ret.model = (dedup.length === 1 ? dedup[0] : dedup);

        ret.architecture = os.arch();
    }
    ret.logicalCores = cpus.length;
    if (process.platform !== 'win32') {
        ret.utilization = formatUtil(os.loadavg()[0]);
    }
    return ret;
}

async function getDiskInfo(): Promise<any> {
    // FIXME: @types/node is outdated
    const statfs: any = await (fs as any).statfs(globals.paths.experimentRoot);
    const typeHex = '0x' + statfs.type.toString(16);
    return {
        filesystem: fsTypes[typeHex] ?? typeHex,
        space: formatSize(statfs.blocks * statfs.bsize),
        availableSpace: formatSize(statfs.bavail * statfs.bsize),
        utilization: formatUtil(1 - statfs.bavail / statfs.blocks),
    };
}

const fsTypes: Record<string, string> = {  // https://man7.org/linux/man-pages/man2/statfs.2.html
    '0x9123683e': 'btrfs',
    '0xef53': 'ext4',
    '0x65735546': 'fuse',
    '0x6969': 'nfs',
    '0x6e736673': 'ntfs',
    '0x01021994': 'tmpfs',
};

function formatUtil(util: number): string {
    return `${Math.round(util * 100)}%`;
}

function formatSize(size: number, disk?: boolean): string {
    let units;
    if (disk) {
        units = [ 'KiB', 'MiB', 'GiB', 'TiB' ];
    } else {
        units = [ 'KB', 'MB', 'GB' ];
    }

    let num = size;
    let unit = 'B';
    for (const u of units) {
        if (num >= 1024) {
            num /= 1024;
            unit = u;
        } else {
            break;
        }
    }

    if (num >= 10) {
        num = Math.round(num);
    } else {
        num = Math.round(num * 10) / 10;
    }

    return `${num}${unit}`;
}
