const batch_threshold = 0.5;

/**
 *  Format objects in a "batch", which means they should either be all collapsed or all expanded.
 *
 *  Note that in this file the word "object" means any serializable value,
 *  while JavaScript `Object` is called "dict" instead.
 *
 *  Caller should use `detect_batch` to ensure that all non-null `objects` are "batchable".
 *  A single object is always a valid batch, and `null` can be batched with any value.
 *
 *  If the objects are values of dict, their keys can be passed with `key_or_keys`,
 *  which will add `"key": ` before each stringified object.
 *
 *  @param objects  Objects to be stringify.
 *  @param indent  Spaces that should be prepended to each line.
 *  @param width  Expected width of text block. This is only a hint, not hard limit.
 *  @param key_or_keys  Array of keys for each object,
 *      or a single string as the same key of all objects,
 *      or `undefined` if they are not dict value.
 *
 *  @returns  Formatted string for each object, without trailing comma.
 **/
function batch_format(objects: any[], indent: string, width: number, key_or_keys?: string | string[]): string[] {
    let keys: string[];  // dict key as prefix string
    if (key_or_keys === undefined) {
        keys = objects.map(() => '');
    } else if (typeof key_or_keys === 'string') {
        keys = objects.map(() => `"${key_or_keys}": `);
    } else {
        keys = key_or_keys.map(k => `"${k}": `);
    }

    // try to collapse all
    const lines = objects.map((obj, i) => keys[i] + stringify_single_line(obj));
    if (lines.every(line => (line.length + indent.length < width))) {
        return lines;
    }

    // null values don't affect hierarchy detection
    const non_null = objects.filter(obj => obj !== null);
    if (non_null.length === 0) {
        return lines;
    }

    if (Array.isArray(non_null[0])) {
        // objects are arrays, format all items in one batch
        const elements = batch_format(concat(non_null), indent + '    ', width);
        const iter = elements[Symbol.iterator]();
        return objects.map((obj, i) => {
            if (obj === null) {
                return keys[i] + 'null';
            } else {
                return keys[i] + create_block(indent, '[]', obj.map(() => iter.next().value));
            }
        });
    }

    if (typeof non_null[0] === 'object') {
        // objects are dicts, format values in one batch if they have similar keys
        const values = concat(non_null.map(obj => Object.values(obj)));
        const children_keys = concat(non_null.map(obj => Object.keys(obj)));
        if (detect_batch(values)) {
            // these objects look like TypeScript style `Map` or `Record`, where the values have same "type"
            const elements = batch_format(values, indent + '    ', width, children_keys);
            const iter = elements[Symbol.iterator]();
            return objects.map((obj, i) => {
                if (obj === null) {
                    return keys[i] + 'null';
                } else {
                    return keys[i] + create_block(indent, '{}', Object.keys(obj).map(() => iter.next().value));
                }
            });

        } else {
            // these objects look like class instances, so we will try to group their fields
            const unique_keys = new Set(children_keys);
            const iters = new Map();
            for (let key of unique_keys) {
                const fields = non_null.map(obj => obj[key]).filter(v => v !== undefined);
                let elements;
                if (detect_batch(fields)) {  // look like same field of class instances
                    elements = batch_format(fields, indent + '    ', width, key);
                } else {  // no idea what these are, fallback to format them independently
                    elements = fields.map(field => batch_format([field], indent + '    ', width, key));
                }
                iters.set(key, elements[Symbol.iterator]());
            }
            return objects.map((obj, i) => {
                if (obj === null) {
                    return keys[i] + 'null';
                } else {
                    const elements = Object.keys(obj).map(key => iters.get(key).next().value);
                    return keys[i] + create_block(indent, '{}', elements);
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
 *  The threshold of similarity is defined by `batch_threshold`, which is 0.5 by default.
 *  Dicts are considered batchable iff their similarity value is greater than the threshold.
 *
 *  @param objects  The objects to be analyzed.
 *
 *  @returns  `true` if objects should be batched; `false` otherwise.
 **/
function detect_batch(objects: any[]): boolean {
    const non_null = objects.filter(obj => obj !== null);

    if (non_null.every(obj => Array.isArray(obj))) {
        return same_type(concat(non_null));
    }

    if (non_null.every(obj => (typeof obj === 'object' && !Array.isArray(obj)))) {
        const total_keys = new Set(concat(non_null.map(obj => Object.keys(obj)))).size;
        const miss_keys = non_null.map(obj => (total_keys - Object.keys(obj).length));
        const miss_sum = miss_keys.reduce((a, b) => a + b, 0);
        return miss_sum < (total_keys * non_null.length) * batch_threshold;
    }

    return same_type(non_null);
}

function concat(arrays: any[][]): any[] {
    return ([] as any[]).concat(...arrays);
}

function create_block(indent: string, brackets: string, elements: string[]): string {
    if (elements.length === 0) {
        return brackets;
    }
    const head = brackets[0] + '\n    ' + indent;
    const line_separator = ',\n    ' + indent;
    const tail = '\n' + indent + brackets[1];
    return head + elements.join(line_separator) + tail;
}

function same_type(objects: any[]) {
    const non_null = objects.filter(obj => obj !== undefined);
    return non_null.length > 0 ? non_null.every(obj => (typeof obj === typeof non_null[0])) : true;
}

function stringify_single_line(obj: any) {
    if (obj === null) {
        return 'null';
    } else if (typeof obj === 'number' || typeof obj === 'boolean') {
        return obj.toString();
    } else if (typeof obj === 'string') {
        return `"${obj}"`;
    } else if (Array.isArray(obj)) {
        return '[' + obj.map(x => stringify_single_line(x)).join(', ') + ']'
    } else {
        return '{' + Object.keys(obj).map(key => `"${key}": ${stringify_single_line(obj[key])}`).join(', ') + '}';
    }
}

function pretty_stringify(obj: any, width: number): string {
    return batch_format([obj], '', width)[0];
}

export { pretty_stringify };
