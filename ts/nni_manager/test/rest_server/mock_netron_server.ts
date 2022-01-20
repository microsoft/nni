// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  A helper HTTP server that sends request informantion back as response.
 *  Used to check that RestServer will send correct requests to netron.app.
 *
 *  It will automatically dispose itself so no need for stop().
 **/

import type { AddressInfo } from 'net';

import express, { Request, Response } from 'express';
import { Deferred } from 'ts-deferred';

export function start(): Promise<number> {
    const app = express();
    app.use(express.text());

    app.get('*', (req: Request, res: Response) => {
        res.send({
            method: 'GET',
            url: req.url,
            headers: req.headers,
        });
    });

    app.post('*', (req: Request, res: Response) => {
        res.send({
            method: 'POST',
            url: req.url,
            headers: req.headers,
            body: req.body,
        });
    });

    const server = app.listen();
    server.unref();
    const deferred = new Deferred<number>();
    server.on('listening', () => {
        deferred.resolve((<AddressInfo>server!.address()).port);
    });
    return deferred.promise;
}
