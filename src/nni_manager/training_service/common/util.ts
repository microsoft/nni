import { getLogger } from "common/log";

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

import { countFilesRecursively } from '../../common/utils'

/**
 * Validate codeDir, calculate file count recursively under codeDir, and throw error if any rule is broken
 * 
 * @param codeDir codeDir in nni config file
 * @returns file number under codeDir
 */
export async function validateCodeDir(codeDir: string) : Promise<number> {
    let fileCount: number | undefined;

    try {
        fileCount = await countFilesRecursively(codeDir);
    } catch(error) {
        throw new Error(`Call count file error: ${error}`);
    }

    if(fileCount && fileCount > 1000) {
        const errMessage: string = `Too many files(${fileCount} found}) in ${codeDir},` 
                                    + ` please check if it's a valid code dir`;
        throw new Error(errMessage);        
    }

    return fileCount;
}