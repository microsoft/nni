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

import { getBasePort, getExperimentId } from '../../common/experimentStartupInfo';

import * as assert from 'assert';
import { getLogger, Logger } from '../../common/log';

const syslogd = require('syslogd');

class SysLogServer {
    private readonly sysLogServer: any;
    private readonly port: number;
    private readonly log!: Logger;

    constructor() {
        this.log = getLogger();
        this.sysLogServer  = syslogd((data: any) => {
            this.log.info(`Syslog get data: ${JSON.stringify(data)}`);
        });
        const basePort: number = getBasePort();
        assert(basePort && basePort > 1024);
        
        this.port = basePort + 2;
    }

    public start() {
        return;
        // this.sysLogServer.listen(this.port, (err: any) => {
        //     if(err) {
        //       this.log.error(err);
        //       return;
        //     }
        //     this.log.info(`listen on ${this.port}`);
        //    });
    }

    public get sysLogPort(): number {
        return this.port;
    }
}

export { SysLogServer }