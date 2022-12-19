export interface StorageFormat {
    expire: number;
    time: number;
    value: string;
}

export const getValue = (key): null | string => {
    const val = localStorage.getItem(key);
    if (!val) {
        return null;
    }
    const data = JSON.parse(val) as StorageFormat;
    if (Date.now() - data.time > data.expire) {
        localStorage.removeItem(key);
        return null;
    }
    return data.value;
};

class Storage {
    key: string = '';
    value: string = '';
    expire: number = 0;

    constructor(key: string, value: string, expire: number) {
        this.key = key;
        this.value = value;
        this.expire = expire;
    }

    public setValue(): void {
        const obj: StorageFormat = {
            value: this.value,
            time: Date.now(),
            expire: this.expire
        };
        localStorage.setItem(this.key, JSON.stringify(obj));
    }
}

export { Storage };
