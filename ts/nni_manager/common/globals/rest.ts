// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WIP REST API manager.
 *  Functions will be added when used.
 **/

import express, { Request, Response, Router } from 'express';

type HttpMethod = 'GET' | 'PUT';

type ExpressCallback = (req: Request, res: Response) => void;

export class RestManager {
    private router: Router;

    constructor() {
        this.router = Router();
        this.router.use(express.json({ limit: '50mb' }));
    }

    public getExpressRouter(): Router {
        return this.router;
    }

    public registerSyncHandler(method: HttpMethod, path: string, callback: ExpressCallback): void {
        const p = '/' + trimSlash(path);
        if (method === 'GET') {
            this.router.get(p, callback);
        } else if (method === 'PUT') {
            this.router.put(p, callback);
        } else {
            throw new Error(`RestManager: Bad method ${method}`);
        }
    }

    public registerExpressRouter(path: string, router: Router): void {
        this.router.use(path, router);
    }

    public urlJoin(...parts: string[]): string {
        return parts.map(trimSlash).filter(part => part).join('/');
    }

    public getFullUrl(protocol: string, ...parts: string[]): string {
        const root = `${protocol}://localhost:${global.nni.args.port}/`;
        return root + this.urlJoin(global.nni.args.urlPrefix, ...parts);
    }
}

function trimSlash(s: string): string {
    return s.replace(/^\/+|\/+$/g, '');
}
