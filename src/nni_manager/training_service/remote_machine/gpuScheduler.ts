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

import * as assert from 'assert';
import { getLogger, Logger } from '../../common/log';
import { TrialJobDetail } from '../../common/trainingService';
import { randomSelect } from '../../common/utils';
import { GPUInfo } from '../common/gpuData';
import {
    parseGpuIndices, RemoteMachineMeta, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail, ScheduleResultType, SSHClientManager
} from './remoteMachineData';

/**
 * A simple GPU scheduler implementation
 */
export class GPUScheduler {

    private readonly machineSSHClientMap : Map<RemoteMachineMeta, SSHClientManager>;
    private readonly log: Logger = getLogger();

    /**
     * Constructor
     * @param machineSSHClientMap map from remote machine to sshClient
     */
    constructor(machineSSHClientMap : Map<RemoteMachineMeta, SSHClientManager>) {
        this.machineSSHClientMap = machineSSHClientMap;
    }

    /**
     * Schedule a machine according to the constraints (requiredGPUNum)
     * @param requiredGPUNum required GPU number
     */
    public scheduleMachine(requiredGPUNum: number, trialJobDetail : RemoteMachineTrialJobDetail) : RemoteMachineScheduleResult {
        assert(requiredGPUNum >= 0);
        const allRMs: RemoteMachineMeta[] = Array.from(this.machineSSHClientMap.keys());
        assert(allRMs.length > 0);

        // Step 1: Check if required GPU number not exceeds the total GPU number in all machines
        const eligibleRM: RemoteMachineMeta[] = allRMs.filter((rmMeta : RemoteMachineMeta) =>
                 rmMeta.gpuSummary === undefined || requiredGPUNum === 0 || rmMeta.gpuSummary.gpuCount >= requiredGPUNum);
        if (eligibleRM.length === 0) {
            // If the required gpu number exceeds the upper limit of all machine's GPU number
            // Return REQUIRE_EXCEED_TOTAL directly
            return ({
                resultType: ScheduleResultType.REQUIRE_EXCEED_TOTAL,
                scheduleInfo: undefined
            });
        }

        // Step 2: Allocate Host/GPU for specified trial job
        // Currenty the requireGPUNum parameter for all trial jobs are identical.
        if (requiredGPUNum > 0) {
            // Trial job requires GPU
            const result: RemoteMachineScheduleResult | undefined = this.scheduleGPUHost(requiredGPUNum, trialJobDetail);
            if (result !== undefined) {
                return result;
            }
        } else {
            // Trail job does not need GPU
            const allocatedRm: RemoteMachineMeta = this.selectMachine(allRMs);

            return this.allocateHost(requiredGPUNum, allocatedRm, [], trialJobDetail);
        }
        this.log.warning(`Scheduler: trialJob id ${trialJobDetail.id}, no machine can be scheduled, return TMP_NO_AVAILABLE_GPU `);

        return {
            resultType : ScheduleResultType.TMP_NO_AVAILABLE_GPU,
            scheduleInfo : undefined
        };
    }

    /**
     * remove the job's gpu reversion
     */
    public removeGpuReservation(trialJobId: string, trialJobMap: Map<string, RemoteMachineTrialJobDetail>): void {
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = trialJobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`could not get trialJobDetail by id ${trialJobId}`);
        }
        if (trialJobDetail.rmMeta !== undefined &&
            trialJobDetail.rmMeta.occupiedGpuIndexMap !== undefined &&
            trialJobDetail.gpuIndices !== undefined &&
            trialJobDetail.gpuIndices.length > 0) {
            for (const gpuInfo of trialJobDetail.gpuIndices) {
                const num: number | undefined = trialJobDetail.rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                if (num !== undefined) {
                    if (num === 1) {
                        trialJobDetail.rmMeta.occupiedGpuIndexMap.delete(gpuInfo.index);
                    } else {
                        trialJobDetail.rmMeta.occupiedGpuIndexMap.set(gpuInfo.index, num - 1);
                    }
                }
            }
        }
        trialJobDetail.gpuIndices = [];
        trialJobMap.set(trialJobId, trialJobDetail);
    }

    private scheduleGPUHost(requiredGPUNum: number, trialJobDetail: RemoteMachineTrialJobDetail): RemoteMachineScheduleResult | undefined {
        const totalResourceMap: Map<RemoteMachineMeta, GPUInfo[]> = this.gpuResourceDetection();
        const qualifiedRMs: RemoteMachineMeta[] = [];
        totalResourceMap.forEach((gpuInfos: GPUInfo[], rmMeta: RemoteMachineMeta) => {
            if (gpuInfos !== undefined && gpuInfos.length >= requiredGPUNum) {
                qualifiedRMs.push(rmMeta);
            }
        });
        if (qualifiedRMs.length > 0) {
            const allocatedRm: RemoteMachineMeta = this.selectMachine(qualifiedRMs);
            const gpuInfos: GPUInfo[] | undefined = totalResourceMap.get(allocatedRm);
            if (gpuInfos !== undefined) { // should always true
                return this.allocateHost(requiredGPUNum, allocatedRm, gpuInfos, trialJobDetail);
            } else {
                assert(false, 'gpuInfos is undefined');
            }
        }
    }

    /**
     * Detect available GPU resource for a remote machine
     * @param rmMeta Remote machine metadata
     * @param requiredGPUNum required GPU number by application
     * @param availableGPUMap available GPU resource filled by this detection
     * @returns Available GPU number on this remote machine
     */
    private gpuResourceDetection() : Map<RemoteMachineMeta, GPUInfo[]> {
        const totalResourceMap : Map<RemoteMachineMeta, GPUInfo[]> = new Map<RemoteMachineMeta, GPUInfo[]>();
        this.machineSSHClientMap.forEach((sshClientManager: SSHClientManager, rmMeta: RemoteMachineMeta) => {
            // Assgin totoal GPU count as init available GPU number
            if (rmMeta.gpuSummary !== undefined) {
                const availableGPUs: GPUInfo[] = [];
                const designatedGpuIndices: Set<number> | undefined = parseGpuIndices(rmMeta.gpuIndices);
                if (designatedGpuIndices !== undefined) {
                    for (const gpuIndex of designatedGpuIndices) {
                        if (gpuIndex >= rmMeta.gpuSummary.gpuCount) {
                            throw new Error(`Specified GPU index not found: ${gpuIndex}`);
                        }
                    }
                }
                this.log.debug(`designated gpu indices: ${designatedGpuIndices}`);
                // tslint:disable: strict-boolean-expressions
                rmMeta.gpuSummary.gpuInfos.forEach((gpuInfo: GPUInfo) => {
                    // if the GPU has active process, OR be reserved by a job,
                    // or index not in gpuIndices configuration in machineList,
                    // or trial number on a GPU reach max number,
                    // We should NOT allocate this GPU
                    // if users set useActiveGpu, use the gpu whether there is another activeProcess
                    if (designatedGpuIndices === undefined || designatedGpuIndices.has(gpuInfo.index)) {
                        if (rmMeta.occupiedGpuIndexMap !== undefined) {
                            const num: number | undefined = rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                            const maxTrialNumPerGpu: number = rmMeta.maxTrialNumPerGpu ? rmMeta.maxTrialNumPerGpu : 1;
                            if ((num === undefined && (!rmMeta.useActiveGpu && gpuInfo.activeProcessNum === 0 || rmMeta.useActiveGpu)) ||
                               (num !== undefined && num < maxTrialNumPerGpu)) {
                                availableGPUs.push(gpuInfo);
                            }
                        } else {
                            throw new Error(`occupiedGpuIndexMap initialize error!`);
                        }
                    }
                });
                totalResourceMap.set(rmMeta, availableGPUs);
            }
        });

        return totalResourceMap;
    }
    // tslint:enable: strict-boolean-expressions

    private selectMachine(rmMetas: RemoteMachineMeta[]): RemoteMachineMeta {
        assert(rmMetas !== undefined && rmMetas.length > 0);

        return randomSelect(rmMetas);
    }

    private selectGPUsForTrial(gpuInfos: GPUInfo[], requiredGPUNum: number): GPUInfo[] {
        // Sequentially allocate GPUs
        return gpuInfos.slice(0, requiredGPUNum);
    }

    private allocateHost(requiredGPUNum: number, rmMeta: RemoteMachineMeta,
                         gpuInfos: GPUInfo[], trialJobDetail: RemoteMachineTrialJobDetail): RemoteMachineScheduleResult {
        assert(gpuInfos.length >= requiredGPUNum);
        const allocatedGPUs: GPUInfo[] = this.selectGPUsForTrial(gpuInfos, requiredGPUNum);
        allocatedGPUs.forEach((gpuInfo: GPUInfo) => {
            if (rmMeta.occupiedGpuIndexMap !== undefined) {
                let num: number | undefined = rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                if (num === undefined) {
                    num = 0;
                }
                rmMeta.occupiedGpuIndexMap.set(gpuInfo.index, num + 1);
            } else {
                throw new Error(`Machine ${rmMeta.ip} occupiedGpuIndexMap initialize error!`);
            }
        });
        trialJobDetail.gpuIndices = allocatedGPUs;
        trialJobDetail.rmMeta = rmMeta;

        return {
            resultType: ScheduleResultType.SUCCEED,
            scheduleInfo: {
                rmMeta: rmMeta,
                cuda_visible_device: allocatedGPUs
                                       .map((gpuInfo: GPUInfo) => {
                                            return gpuInfo.index;
                                        })
                                       .join(',')
            }
        };
    }
}
