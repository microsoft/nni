// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { RemoteMachineMeta } from '../../remote_machine/remoteMachineData';
import { getLogger, Logger } from '../../../common/log';

/**
 * A simple GPU scheduler implementation
 */
export class MachineScheduler {

    private readonly remoteMachineMetaOccupiedMap: Map<RemoteMachineMeta, boolean>;
    private readonly log: Logger;

    constructor(remoteMachineMetaList: RemoteMachineMeta[]) {
        assert(remoteMachineMetaList.length > 0);
        this.remoteMachineMetaOccupiedMap = new Map<RemoteMachineMeta, boolean>();
        this.log = getLogger();
        this.log.info(`initialize machine scheduler, machine size ${remoteMachineMetaList.length}`);
        remoteMachineMetaList.forEach(rmMeta => {
            // initialize remoteMachineMetaOccupiedMap, false means not occupied
            this.remoteMachineMetaOccupiedMap.set(rmMeta, false);
        });
    }

    public scheduleMachine(): RemoteMachineMeta | undefined {
        for (const [rmMeta, occupied] of this.remoteMachineMetaOccupiedMap) {
            if (!occupied) {
                this.remoteMachineMetaOccupiedMap.set(rmMeta, true);
                return rmMeta;
            }
        }
        return undefined;
    }

    public recycleMachineReservation(rmMeta: RemoteMachineMeta): void {
        if (!this.remoteMachineMetaOccupiedMap.has(rmMeta)) {
            throw new Error(`${rmMeta} not initialized!`);
        }
        this.remoteMachineMetaOccupiedMap.set(rmMeta, false);
    }
}
