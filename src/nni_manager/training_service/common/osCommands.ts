// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

abstract class OsCommands {

    protected pathSpliter: string = '/';
    protected multiplePathSpliter: RegExp = new RegExp(`\\${this.pathSpliter}{2,}`);

    constructor() {
    }

    public abstract createFolder(folderName: string, sharedFolder: boolean): string;
    public abstract removeFolder(folderName: string, isRecursive: boolean, isForce: boolean): string;
    public abstract removeFiles(folderOrFileName: string, filePattern: string, isRecursive: boolean): string;

    public joinPath(...paths: string[]): string {
        let dir: string = paths.filter((path: any) => path !== '').join(this.pathSpliter);
        if (dir === '') {
            dir = '.';
        } else {
            dir = dir.replace(this.multiplePathSpliter, this.pathSpliter);
        }
        return dir;
    }
}

export { OsCommands };
