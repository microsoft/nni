export interface StorageFormat {
    expire: number;
    time: number;
    value: string;
    isSelectall: boolean;
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
    isSelectall: boolean = false;

    constructor(key: string, value: string, expire: number, isSelectall: boolean) {
        this.key = key;
        this.value = value;
        this.expire = expire;
        this.isSelectall = isSelectall;
    }

    public setValue(): void {
        const obj: StorageFormat = {
            value: this.value,
            time: Date.now(),
            expire: this.expire,
            isSelectall: this.isSelectall
        };
        localStorage.setItem(this.key, JSON.stringify(obj));
    }
}

export { Storage };
