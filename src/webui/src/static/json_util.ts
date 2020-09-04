const batchThreshold = 0.5;

/**
 *  Format objects in a "batch", which means they should either be all collapsed or all expanded.
 *
 *  Note that in this file the word "object" means any serializable value,
 *  while JavaScript `Object` is called "dict" instead.
 *
 *  Caller should use `detectBatch` to ensure that all non-null `objects` are "batchable".
 *  A single object is always a valid batch, and `null` can be batched with any value.
 *
 *  If the objects are values of dict, their keys can be passed with `keyOrKeys`,
 *  which will add `"key": ` before each stringified object.
 *
 *  @param objects  Objects to be stringify.
 *  @param indent  Spaces that should be prepended to each line.
 *  @param width  Expected width of text block. This is only a hint, not hard limit.
 *  @param keyOrKeys  Array of keys for each object,
 *      or a single string as the same key of all objects,
 *      or `undefined` if they are not dict value.
 *
 *  @returns  Formatted string for each object, without trailing comma.
 **/
function batchFormat(objects: any[], indent: string, width: number, keyOrKeys?: string | string[]): string[] {
    let keys: string[];  // dict key as prefix string
    if (keyOrKeys === undefined) {
        keys = objects.map(() => '');
    } else if (typeof keyOrKeys === 'string') {
        keys = objects.map(() => `"${keyOrKeys}": `);
    } else {
        keys = keyOrKeys.map(k => `"${k}": `);
    }

    // try to collapse all
    const lines = objects.map((obj, i) => keys[i] + stringifySingleLine(obj));
    if (lines.every(line => (line.length + indent.length < width))) {
        return lines;
    }

    // null values don't affect hierarchy detection
    const nonNull = objects.filter(obj => obj !== null);
    if (nonNull.length === 0) {
        return lines;
    }

    if (Array.isArray(nonNull[0])) {
        // objects are arrays, format all items in one batch
        const elements = batchFormat(concat(nonNull), indent + '    ', width);
        const iter = elements[Symbol.iterator]();
        return objects.map((obj, i) => {
            if (obj === null) {
                return keys[i] + 'null';
            } else {
                return keys[i] + createBlock(indent, '[]', obj.map(() => iter.next().value));
            }
        });
    }

    if (typeof nonNull[0] === 'object') {
        // objects are dicts, format values in one batch if they have similar keys
        const values = concat(nonNull.map(obj => Object.values(obj)));
        const childrenKeys = concat(nonNull.map(obj => Object.keys(obj)));
        if (detectBatch(values)) {
            // these objects look like TypeScript style `Map` or `Record`, where the values have same "type"
            const elements = batchFormat(values, indent + '    ', width, childrenKeys);
            const iter = elements[Symbol.iterator]();
            return objects.map((obj, i) => {
                if (obj === null) {
                    return keys[i] + 'null';
                } else {
                    return keys[i] + createBlock(indent, '{}', Object.keys(obj).map(() => iter.next().value));
                }
            });

        } else {
            // these objects look like class instances, so we will try to group their fields
            const uniqueKeys = new Set(childrenKeys);
            const iters = new Map();
            for (let key of uniqueKeys) {
                const fields = nonNull.map(obj => obj[key]).filter(v => v !== undefined);
                let elements;
                if (detectBatch(fields)) {  // look like same field of class instances
                    elements = batchFormat(fields, indent + '    ', width, key);
                } else {  // no idea what these are, fallback to format them independently
                    elements = fields.map(field => batchFormat([field], indent + '    ', width, key));
                }
                iters.set(key, elements[Symbol.iterator]());
            }
            return objects.map((obj, i) => {
                if (obj === null) {
                    return keys[i] + 'null';
                } else {
                    const elements = Object.keys(obj).map(key => iters.get(key).next().value);
                    return keys[i] + createBlock(indent, '{}', elements);
                }
            });
        }
    }

    // objects are primitive, impossible to break lines although they are too long
    return lines;
}

/**
 *  Detect whether objects should be formated as a batch or formatted on their own.
 *
 *  Objects should be batched if and only if one of following conditions holds:
 *    * They are all primitive values.
 *    * They are all arrays or null.
 *    * They are all dicts or null, and the dicts have similar keys.
 *
 *  For dicts, we assume the perfect situation is that each dict has all keys.
 *  Then we measure their similarity by how many fields are "missing" in order to become perfect match.
 *  The similarity value is calculated as:
 *      number of missing fields / total fields of all dicts if they are perfectly matched
 *  The threshold of similarity is defined by `batchThreshold`, which is 0.5 by default.
 *  Dicts are considered batchable iff their similarity value is greater than the threshold.
 *
 *  @param objects  The objects to be analyzed.
 *
 *  @returns  `true` if objects should be batched; `false` otherwise.
 **/
function detectBatch(objects: any[]): boolean {
    const nonNull = objects.filter(obj => obj !== null);

    if (nonNull.every(obj => Array.isArray(obj))) {
        return sameType(concat(nonNull));
    }

    if (nonNull.every(obj => (typeof obj === 'object' && !Array.isArray(obj)))) {
        const totalKeys = new Set(concat(nonNull.map(obj => Object.keys(obj)))).size;
        const missKeys = nonNull.map(obj => (totalKeys - Object.keys(obj).length));
        const missSum = missKeys.reduce((a, b) => a + b, 0);
        return missSum < (totalKeys * nonNull.length) * batchThreshold;
    }

    return sameType(nonNull);
}

function concat(arrays: any[][]): any[] {
    return ([] as any[]).concat(...arrays);
}

function createBlock(indent: string, brackets: string, elements: string[]): string {
    if (elements.length === 0) {
        return brackets;
    }
    const head = brackets[0] + '\n    ' + indent;
    const lineSeparator = ',\n    ' + indent;
    const tail = '\n' + indent + brackets[1];
    return head + elements.join(lineSeparator) + tail;
}

function sameType(objects: any[]) {
    const nonNull = objects.filter(obj => obj !== undefined);
    return nonNull.length > 0 ? nonNull.every(obj => (typeof obj === typeof nonNull[0])) : true;
}

function stringifySingleLine(obj: any) {
    if (obj === null) {
        return 'null';
    } else if (typeof obj === 'number' || typeof obj === 'boolean') {
        return obj.toString();
    } else if (typeof obj === 'string') {
        return `"${obj}"`;
    } else if (Array.isArray(obj)) {
        return '[' + obj.map(x => stringifySingleLine(x)).join(', ') + ']'
    } else {
        return '{' + Object.keys(obj).map(key => `"${key}": ${stringifySingleLine(obj[key])}`).join(', ') + '}';
    }
}

function prettyStringify(obj: any, width: number): string {
    return batchFormat([obj], '', width)[0];
}

export { prettyStringify };
