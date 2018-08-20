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

import { Client } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { getLogger, Logger } from '../../common/log';
import { GPUInfo } from '../common/gpuData';
import { RemoteMachineMeta, RemoteMachineScheduleResult, RemoteMachineScheduleInfo, ScheduleResultType } from './remoteMachineData';

/**
 * A simple GPU scheduler implementation
 */
export class GPUScheduler {

    private readonly machineSSHClientMap : Map<RemoteMachineMeta, Client>;
    private log: Logger = getLogger();

    /**
     * Constructor
     * @param machineSSHClientMap map from remote machine to sshClient
     */
    constructor(machineSSHClientMap : Map<RemoteMachineMeta, Client>) {
        this.machineSSHClientMap = machineSSHClientMap;
    }

    /**
     * Schedule a machine according to the constraints (requiredGPUNum)
     * @param requiredGPUNum required GPU number
     */
    public scheduleMachine(requiredGPUNum : Number | undefined, trialJobId : string) : RemoteMachineScheduleResult {
        const deferred: Deferred<RemoteMachineScheduleResult> = new Deferred<RemoteMachineScheduleResult>();
        let scheduleResult : RemoteMachineScheduleResult = {
            resultType : ScheduleResultType.TMP_NO_AVAILABLE_GPU,
            scheduleInfo : undefined
        };
        
        // Step 0: Check if required GPU number not exceeds the total GPU number in all machines
        const eligibleRM : RemoteMachineMeta[] = Array.from(this.machineSSHClientMap.keys()).filter((rmMeta : RemoteMachineMeta) =>
                 rmMeta.gpuSummary === undefined || requiredGPUNum === undefined || rmMeta.gpuSummary.gpuCount >= requiredGPUNum );
        if(eligibleRM.length == 0) {
            // If the required gpu number exceeds the upper limit of all machine's GPU number
            // Return REQUIRE_EXCEED_TOTAL directly
            return ({
                resultType : ScheduleResultType.REQUIRE_EXCEED_TOTAL,
                scheduleInfo : undefined
            });
        }

        // Step 1: Generate GPU resource map for remote machines
        const totalResourceMap : Map<RemoteMachineMeta, GPUInfo[]>  = this.gpuResourceDetection(requiredGPUNum);
        
        // Step 2: Find machine whose GPU can be allocated based on user GPU requirement, and allocate GPU
        for (const rmMeta of Array.from(totalResourceMap.keys())) {
            const gpuInfos : GPUInfo[] | undefined = totalResourceMap.get(rmMeta);
            if(gpuInfos !== undefined && (requiredGPUNum === undefined ||  gpuInfos.length >= requiredGPUNum)) {
                const allocatedGPUIndex : number[] = Array();

                // Allocate
                gpuInfos.forEach((gpuInfo : GPUInfo) => {
                    rmMeta.gpuReservation.set(gpuInfo.index, trialJobId);
                    allocatedGPUIndex.push(gpuInfo.index);
                });

                // Construct scheduling return object
                const sshClient : Client | undefined = this.machineSSHClientMap.get(rmMeta);
                if(sshClient !== undefined) {
                    this.log.info(`Found available machine, trialJobId is ${trialJobId}, ip is ${rmMeta.ip}, gpu allocated is ${allocatedGPUIndex.toString()}`);
                    // We found the first available machine whose GPU resource can match user requirement
                    return  {
                        resultType : ScheduleResultType.SUCCEED,
                        scheduleInfo : {
                            rmMeta : rmMeta,
                            client : sshClient,
                            cuda_visible_device : allocatedGPUIndex.join(',')
                        }
                    }; 
                }
            }
        }        
        
        // Step 3: If not found machine whose GPU is availabe, then find the first machine whose GPU summary is unknown
        for (const rmMeta of Array.from(this.machineSSHClientMap.keys())) {        
            const client : Client | undefined = this.machineSSHClientMap.get(rmMeta);
            if(rmMeta.gpuSummary == undefined && client !== undefined) {
                // We found the firstmachine whose GPU summary is unknown
                return {
                    resultType : ScheduleResultType.SUCCEED,
                    scheduleInfo :{
                        rmMeta : rmMeta,
                        client : client,
                        //Since gpu information is unknown, make all GPU resources visible to the job
                        cuda_visible_device : ''
                    }
                };
            }
        };
        
        this.log.warning(`Scheduler: trialJob id ${trialJobId}, no machine can be scheduled, resolve as TMP_NO_AVAILABLE_GPU `);
        // Otherwise, no machine can be scheduled, resolve as TMP_NO_AVAILABLE_GPU 
        return {
            resultType : ScheduleResultType.TMP_NO_AVAILABLE_GPU,
            scheduleInfo : undefined
        };
    }

    /**
     * Detect available GPU resource for a remote machine
     * @param rmMeta Remote machine metadata
     * @param requiredGPUNum required GPU number by application
     * @param availableGPUMap available GPU resource filled by this detection
     * @returns Available GPU number on this remote machine
     */
    private gpuResourceDetection(requiredGPUNum : Number | undefined) : Map<RemoteMachineMeta, GPUInfo[]> {
        const totalResourceMap : Map<RemoteMachineMeta, GPUInfo[]> = new Map<RemoteMachineMeta, GPUInfo[]>();
        this.machineSSHClientMap.forEach((client: Client, rmMeta: RemoteMachineMeta) => {
            // Assgin totoal GPU count as init available GPU number
            if(rmMeta.gpuSummary !== undefined) {
                const availableGPUs : GPUInfo[] = Array();
                if(rmMeta.gpuReservation === undefined) {
                    rmMeta.gpuReservation = new Map<number, string>();
                }
                const gpuReservation = rmMeta.gpuReservation;

                rmMeta.gpuSummary.gpuInfos.forEach((gpuInfo: GPUInfo) => {
                    //this.log.info(`GPU index:${gpuInfo.index}, activeProcessNum is ${gpuInfo.activeProcessNum}, GPU reservation is ${JSON.stringify([...gpuReservation])}`);
                    // if the GPU has active process, OR be reserved by a job, 
                    // We should NOT allocate this GPU
                    if (gpuInfo.activeProcessNum === 0
                        && !gpuReservation.has(gpuInfo.index)
                        && requiredGPUNum !== undefined
                        && availableGPUs.length < requiredGPUNum) {
                        availableGPUs.push(gpuInfo);
                    }
                });

                totalResourceMap.set(rmMeta, availableGPUs);
            }
        });

        return totalResourceMap;
    }
}
