// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as rx from 'rx';
import * as component from '../common/component';

@component.Singleton
class ObservableTimer {
    private observableSource: rx.Observable<number>;
    constructor() {
        // TODO: move 100 and 1000 into constants class
        this.observableSource = rx.Observable.timer(100, 1000).takeWhile(() => true);
    }

    public subscribe(onNext?: (value: any) => void, onError?: (exception: any) => void, onCompleted?: () => void): Rx.IDisposable {
        return this.observableSource.subscribe(onNext, onError, onCompleted);
    }

    public unsubscribe( subscription: Rx.IDisposable): void {
        if(typeof subscription !== 'undefined') {
            subscription.dispose();
        }
    }
}

export { ObservableTimer };
